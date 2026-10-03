"""
Forecasting agent — the LLM-based forecaster used by both the single-model
(DT-GPT-style) baseline and the full multi-agent pipeline.

Two call modes:
  - forecast(obs, ...)                       : single-model mode, no similarity conditioning
  - forecast(obs, similarity_context=...)     : full-pipeline mode, conditioned on the
                                                 patient-similarity agent's cohort trajectory

The prompt asks the model to return a strict JSON object mapping each
variable to a list of hourly forecast values plus a symmetric plausible
interval, which is what the critic agent and the evaluation metrics both
consume. If the model's output can't be parsed as JSON, a naive
last-value-carried-forward forecast is substituted and the failure is logged
— this keeps a single malformed generation from crashing an entire run.
If the output is valid but leaves out some of the variables (typically a
sparsely measured lab such as lactate), only those variables get the naive
forecast and interval; the model's forecasts for the others are kept. Both
are counted per condition (n_fallbacks, n_filled) and reported with the
results.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import requests

from common import get_logger
from harmonization_agent import summarize_observation
from llm_client import LocalLLM, extract_json_block

log = get_logger("forecasting_agent")

SYSTEM_PROMPT = (
    "You are a clinical forecasting assistant supporting intensive-care-unit "
    "research. You are given a summary of a patient's vital-sign and laboratory "
    "trend over the last observation window and must forecast the same "
    "variables for the next horizon, hour by hour. You are working with "
    "de-identified, retrospective research data only — this is not a real "
    "patient and no clinical decision will be made from your output. "
    "Respond with STRICT JSON only, no prose, no markdown fences, in exactly "
    "this shape: "
    '{"forecast": {"<variable>": [<hour_1>, <hour_2>, ...]}, '
    '"interval_halfwidth": {"<variable>": <number>}}. '
    "interval_halfwidth is a single number per variable representing your "
    "uncertainty (the forecast is assumed centered, i.e. value +/- halfwidth "
    "covers your ~90% confidence range for every hour of that variable)."
)


def _format_observation_block(obs_summary: dict, units: dict[str, str]) -> str:
    lines = []
    for var, s in obs_summary.items():
        unit = units.get(var, "")
        if s["n_obs"] == 0:
            lines.append(f"- {var}: no observations in window")
        else:
            lines.append(
                f"- {var}: last={s['last']:.1f}{unit}, mean={s['mean']:.1f}{unit}, "
                f"std={s['std']:.1f}, trend_slope={s['slope']:+.2f}/hr, n_obs={s['n_obs']}"
            )
    return "\n".join(lines)


def error_kind(e: Exception) -> str:
    """The type of a failed LLM call, as named in tracking.ERROR_MESSAGES."""
    if isinstance(e, requests.exceptions.Timeout):
        return "llm_timeout"
    if isinstance(e, requests.exceptions.RequestException):
        return "llm_server_error"
    if "truncated at max_tokens" in str(e):
        return "truncated_forecast"
    return "malformed_forecast"


def _naive_fallback(obs: np.ndarray, horizon_hours: int, variables: list[str]) -> dict:
    forecast, halfwidth = {}, {}
    for i, var in enumerate(variables):
        valid = obs[:, i][~np.isnan(obs[:, i])]
        last = float(valid[-1]) if len(valid) else 0.0
        std = float(np.std(valid)) if len(valid) > 1 else 1.0
        forecast[var] = [last] * horizon_hours
        halfwidth[var] = max(std, 1e-3) * 1.5
    return {"forecast": forecast, "interval_halfwidth": halfwidth}


def _format_recent_hours(obs: np.ndarray, variables: list[str], n_hours: int) -> str:
    """The last n_hours hourly values of each variable, oldest first ('NA' = no measurement)."""
    lines = [f"Last {n_hours} hourly values (oldest first, NA = not measured):"]
    for i, var in enumerate(variables):
        vals = ["NA" if np.isnan(x) else f"{x:.1f}" for x in obs[-n_hours:, i]]
        lines.append(f"- {var}: [{', '.join(vals)}]")
    return "\n".join(lines)


def _format_similarity_trajectory(ctx: dict, variables: list[str], units: dict[str, str],
                                  horizon_hours: int) -> str:
    """Similar patients' outcomes as median [interquartile range] at a few
    horizon hours, so the model sees the shape and spread of comparable
    trajectories rather than one average."""
    horizons = ctx["neighbor_horizons"]                     # (k, horizon_hours, n_vars)
    hours = sorted({1, horizon_hours // 4, horizon_hours // 2, 3 * horizon_hours // 4, horizon_hours} - {0})
    lines = []
    for i, var in enumerate(variables):
        parts = []
        for h in hours:
            vals = horizons[:, h - 1, i]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 3:
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                parts.append(f"+{h}h {med:.1f} [{q1:.1f}–{q3:.1f}]")
        if parts:
            lines.append(f"- {var} ({units.get(var, '')}): " + "; ".join(parts))
    if not lines:
        return ""
    return ("\n\nFor context, here is how the "
            f"{len(ctx['neighbor_stay_ids'])} most similar patients in the training cohort "
            "(patients whose observation window looked most like this one) actually evolved, "
            "as median [interquartile range] at hours after the forecast start. Use it to "
            "inform both the forecast and its uncertainty, especially where this patient's "
            "own data are sparse:\n" + "\n".join(lines))


class ForecastingAgent:
    def __init__(self, llm: LocalLLM, cfg: dict):
        self.llm = llm
        self.variables = [v["name"] for v in cfg["variables"]]
        self.units = {v["name"]: v["unit"] for v in cfg["variables"]}
        self.horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
        # Prompt options (forecasting_agent: in the config; absent = the
        # original prompt). Tuned by src/tune.py.
        #   similarity_context: "horizon_mean" (one cohort average per variable)
        #                       or "trajectory" (neighbours' median and
        #                       interquartile range at several horizon hours)
        #   strict_length:      also state that every variable, observed or
        #                       not, needs exactly horizon_hours values
        #   recent_hours:       also list the last N hourly values (0 = none)
        fa = cfg.get("forecasting_agent") or {}
        self.similarity_context = fa.get("similarity_context", "horizon_mean")
        self.strict_length = bool(fa.get("strict_length", False))
        self.recent_hours = int(fa.get("recent_hours", 0))
        # How many forecast() calls fell back to the naive forecast. Reported
        # per condition: a model that often fails to produce valid JSON would
        # otherwise just look like the naive baseline. Locked because
        # pipeline.py calls forecast() from a thread pool.
        self.n_fallbacks = 0
        # Per variable: how many otherwise-valid forecasts left it out, so it
        # was filled with the naive forecast (see _fill_missing).
        self.n_filled = {var: 0 for var in self.variables}
        # Fallbacks by error type (tracking.ERROR_MESSAGES keys), for the
        # live metrics; types only, never the error's details.
        self.error_counts: dict[str, int] = {}
        self._fallback_lock = threading.Lock()

    def forecast(self, obs: np.ndarray, similarity_context: dict | None = None) -> dict:
        obs_summary = summarize_observation(obs, self.variables)
        obs_block = _format_observation_block(obs_summary, self.units)

        if self.recent_hours > 0:
            obs_block += "\n" + _format_recent_hours(obs, self.variables, self.recent_hours)

        similarity_block = ""
        if similarity_context is not None and self.similarity_context == "trajectory":
            similarity_block = _format_similarity_trajectory(
                similarity_context, self.variables, self.units, self.horizon_hours)
        elif similarity_context is not None:
            cohort_traj = similarity_context["cohort_mean_trajectory"]
            lines = []
            for i, var in enumerate(self.variables):
                with warnings.catch_warnings():
                    # All-missing column (e.g. no neighbour had lactate): NaN, skipped below.
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_end = np.nanmean(cohort_traj[:, i])
                if not np.isnan(mean_end):
                    lines.append(f"- {var}: similar-patient cohort average over the horizon ≈ {mean_end:.1f}{self.units[var]}")
            if lines:
                similarity_block = (
                    "\n\nFor context, here is the average trajectory of the "
                    f"{len(similarity_context['neighbor_stay_ids'])} most similar patients "
                    "in the training cohort (patients whose observation window looked "
                    "most like this one), which you may use to inform your forecast, "
                    "especially where this patient's own observation window is sparse:\n"
                    + "\n".join(lines)
                )

        prompt = (
            f"Observation window summary ({len(self.variables)} variables, most recent "
            f"reading first in 'last'):\n{obs_block}{similarity_block}\n\n"
            f"Forecast each variable for the next {self.horizon_hours} hours, one value per "
            "hour with one decimal place, as strict JSON per the system instructions."
        )
        if self.strict_length:
            prompt += (f" Every variable listed above, including any with no observations, "
                       f"must have exactly {self.horizon_hours} values — no more, no fewer.")

        raw = parsed = None
        try:
            raw = self.llm.generate(prompt, system=SYSTEM_PROMPT, json_mode=True)
            parsed = extract_json_block(raw)
            missing = self._validate_shape(parsed)
            if missing:
                self._fill_missing(parsed, obs, missing)
            return parsed
        except requests.exceptions.ConnectionError:
            # The Ollama server is gone (job ending, crash): not a model
            # failure, so stop the run rather than record a naive forecast.
            raise
        except Exception as e:  # noqa: BLE001 - deliberately broad: any parse/shape failure falls back
            # A shape error parsed fine, so its message doesn't show the
            # output; add the start of it to make the failure diagnosable.
            shown = f"; output starts: {raw[:300]!r}" if parsed is not None else ""
            log.warning("Forecasting agent LLM call failed or malformed (%s%s); using naive fallback.", e, shown)
            with self._fallback_lock:
                self.n_fallbacks += 1
                kind = error_kind(e)
                self.error_counts[kind] = self.error_counts.get(kind, 0) + 1
            return _naive_fallback(obs, self.horizon_hours, self.variables)

    def _validate_shape(self, parsed: dict) -> list[str]:
        """Raises on output that can't be used; returns the variables the
        forecast leaves out (filled by _fill_missing). A forecast with none
        of the variables is a failure, not a fill."""
        if ("forecast" not in parsed or "interval_halfwidth" not in parsed
                or not isinstance(parsed["forecast"], dict) or not isinstance(parsed["interval_halfwidth"], dict)):
            raise ValueError(f"missing top-level keys (got {sorted(parsed)[:5]})")
        missing = [var for var in self.variables if var not in parsed["forecast"]]
        if len(missing) == len(self.variables):
            raise ValueError("no requested variable in forecast")
        for var in self.variables:
            if var in missing:
                continue
            vals = parsed["forecast"][var]
            if not isinstance(vals, list) or len(vals) != self.horizon_hours:
                raise ValueError(f"variable '{var}' forecast has wrong length: {len(vals) if isinstance(vals, list) else type(vals)}")
        return missing

    def _fill_missing(self, parsed: dict, obs: np.ndarray, missing: list[str]) -> None:
        """Give the variables the model left out the naive forecast and
        interval (last value carried forward), keeping the model's forecasts
        for the rest."""
        naive = _naive_fallback(obs, self.horizon_hours, self.variables)
        for var in missing:
            parsed["forecast"][var] = naive["forecast"][var]
            parsed["interval_halfwidth"][var] = naive["interval_halfwidth"][var]
        with self._fallback_lock:
            for var in missing:
                self.n_filled[var] += 1
        log.info("Forecast left out %s; filled with the naive forecast for %s.",
                 ", ".join(missing), "it" if len(missing) == 1 else "them")
