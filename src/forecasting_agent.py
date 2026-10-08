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
consume. The same shape is passed to Ollama as a JSON schema, so decoding
itself is held to exactly forecast_horizon_hours values per variable. If the model's output can't be parsed as JSON, a naive
last-value-carried-forward forecast is substituted and the failure is logged
— this keeps a single malformed generation from crashing an entire run.
If the output is valid but leaves out some of the variables (typically a
sparsely measured lab such as lactate), only those variables get the naive
forecast and interval; the model's forecasts for the others are kept. Both
are counted per condition (n_fallbacks, n_filled) and reported with the
results. A variable the patient has no observation of is filled with the
training cohort's median (reference stats from pipeline.fit_models), not 0.
A variable whose forecast holds a non-finite or absurd value (far outside
its valid range; Med42 once returned -3e9) is treated as left out and filled
the same way; every forecast is clipped to the variables' hard (physically
possible) ranges, e.g. SpO2 <= 100 %.

Prompt and output options (the `forecasting_agent:` config section, tuned by
src/tune.py; each default reproduces the original prompt):
  trend_hint      slope   : last, mean, std and trend_slope per variable
                  none    : no trend_slope
                  damped  : trend_slope, plus a note that ICU vital signs
                            revert toward the mean rather than continue a trend
                  (Trillium lean run: single-model forecasts continued the
                  slope ~1:1 for 24 h while the patients reverted to the mean.)
  drift_damping, level_anchor_weight, level_window_hours: post-processing
                  of the forecast, applied by pipeline.py (postprocess_forecast;
                  defaults 1.0, 0.0, 6 = none). calibrate.py can fit them per
                  condition on the calibration patients (fit_postprocess).
  cohort_anchor   false   : true adds the training cohort's median [IQR] of
                            every variable to the prompt
  interval        constant: one interval half-width per variable
                  per_hour: one per hour (24 more numbers per variable)
                  endpoints: one at the first and one at the last hour,
                            interpolated in between
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import requests

from common import get_logger, hard_ranges, valid_ranges
from harmonization_agent import naive_fill, summarize_observation
from llm_client import LocalLLM, extract_json_block

log = get_logger("forecasting_agent")

TREND_HINTS = ("slope", "none", "damped")
INTERVAL_MODES = ("constant", "per_hour", "endpoints")

_SYSTEM_PROMPT_START = (
    "You are a clinical forecasting assistant supporting intensive-care-unit "
    "research. You are given a summary of a patient's vital-sign and laboratory "
    "trend over the last observation window and must forecast the same "
    "variables for the next horizon, hour by hour. You are working with "
    "de-identified, retrospective research data only — this is not a real "
    "patient and no clinical decision will be made from your output. "
    "Respond with STRICT JSON only, no prose, no markdown fences, in exactly "
    "this shape: "
    '{"forecast": {"<variable>": [<hour_1>, <hour_2>, ...]}, '
)
_INTERVAL_INSTRUCTIONS = {
    "constant": (
        '"interval_halfwidth": {"<variable>": <number>}}. '
        "interval_halfwidth is a single number per variable representing your "
        "uncertainty (the forecast is assumed centered, i.e. value +/- halfwidth "
        "covers your ~90% confidence range for every hour of that variable)."),
    "per_hour": (
        '"interval_halfwidth": {"<variable>": [<hour_1>, <hour_2>, ...]}}. '
        "interval_halfwidth gives, for every hour, the half-width of your ~90% "
        "confidence range around that hour's forecast (value +/- halfwidth). "
        "Uncertainty usually grows the further ahead the hour is."),
    "endpoints": (
        '"interval_halfwidth": {"<variable>": {"start": <number>, "end": <number>}}}. '
        "interval_halfwidth gives the half-width of your ~90% confidence range "
        "around the forecast (value +/- halfwidth) at the first hour (start) and "
        "at the last hour (end); the hours in between are interpolated. "
        "Uncertainty usually grows the further ahead the hour is."),
}


POSTPROCESS_DEFAULTS = {"drift_damping": 1.0, "level_anchor_weight": 0.0, "level_window_hours": 6}


def postprocess_params(cfg: dict) -> dict:
    """The forecasting_agent section's post-processing settings (defaults: none)."""
    fa = cfg.get("forecasting_agent") or {}
    params = {k: type(d)(fa.get(k, d)) for k, d in POSTPROCESS_DEFAULTS.items()}
    for k in ("drift_damping", "level_anchor_weight"):
        if not 0.0 <= params[k] <= 1.0:
            raise ValueError(f"forecasting_agent.{k} must be in [0, 1], not {params[k]}")
    if params["level_window_hours"] < 1:
        raise ValueError("forecasting_agent.level_window_hours must be at least 1")
    return params


def postprocess_arrays(forecast: np.ndarray, obs: np.ndarray, params: dict, hard: np.ndarray) -> np.ndarray:
    """forecast (..., horizon, n_vars) after post-processing, for obs
    (..., obs_hours, n_vars):
      1. level anchor: shift the whole forecast so hour 1 moves a share w
         (level_anchor_weight) of the way from the model's hour 1 — in practice
         the last reading, a single noisy value — to the mean of the last
         level_window_hours observed hours (no observation there: no shift);
      2. drift damping: each hour's change from hour 1 times lambda;
      3. clip to the hard ranges (hard: (n_vars, 2)).
    Vectorised so calibrate.py can fit the parameters on saved forecasts."""
    out = np.array(forecast, dtype=float)
    w, lam = params["level_anchor_weight"], params["drift_damping"]
    if w > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # no observation in the window
            recent = np.nanmean(obs[..., -int(params["level_window_hours"]):, :], axis=-2)
        shift = w * (recent - out[..., 0, :])
        out = out + np.nan_to_num(shift)[..., None, :]
    if lam < 1:
        out = out[..., :1, :] + lam * (out - out[..., :1, :])
    return np.clip(out, hard[:, 0], hard[:, 1])


def postprocess_forecast(result: dict, obs: np.ndarray, variables: list[str], params: dict,
                         hard: np.ndarray) -> dict:
    """postprocess_arrays on one forecast dict (interval half-widths unchanged)."""
    arr = np.array([result["forecast"][var] for var in variables], dtype=float).T   # (horizon, n_vars)
    arr = postprocess_arrays(arr, obs, params, hard)
    return {**result, "forecast": {var: arr[:, i].tolist() for i, var in enumerate(variables)}}


def system_prompt(interval: str = "constant") -> str:
    return _SYSTEM_PROMPT_START + _INTERVAL_INSTRUCTIONS[interval]


SYSTEM_PROMPT = system_prompt("constant")   # the original prompt

DAMPED_TREND_NOTE = (
    "Note on trends: over the next day, ICU vital signs and labs usually stay "
    "near their current level or drift back toward the patient's recent "
    "average. A trend seen in the observation window rarely continues at the "
    "same rate for 24 hours, so do not extend trend_slope in a straight line."
)


def _format_observation_block(obs_summary: dict, units: dict[str, str], trend: bool = True,
                              reference: dict[str, dict] | None = None) -> str:
    """One line per variable. An unobserved variable is named with the
    cohort's typical value when `reference` is given, so the model has
    something to anchor on (without it, models often forecast 0)."""
    lines = []
    for var, s in obs_summary.items():
        unit = units.get(var, "")
        if s["n_obs"] == 0:
            r = (reference or {}).get(var) or {}
            if r.get("median") is not None:
                lines.append(f"- {var}: not measured in this window; typical value in this ICU cohort: "
                             f"median {r['median']:.1f}{unit} (interquartile range {r['q1']:.1f}–{r['q3']:.1f})")
            else:
                lines.append(f"- {var}: no observations in window")
        else:
            slope = f", trend_slope={s['slope']:+.2f}/hr" if trend else ""
            lines.append(
                f"- {var}: last={s['last']:.1f}{unit}, mean={s['mean']:.1f}{unit}, "
                f"std={s['std']:.1f}{slope}, n_obs={s['n_obs']}"
            )
    return "\n".join(lines)


def _format_cohort_anchor(reference: dict[str, dict], variables: list[str], units: dict[str, str]) -> str:
    """The training cohort's median [IQR] per variable (cohort_anchor)."""
    lines = [f"- {var}: {r['median']:.1f} [{r['q1']:.1f}–{r['q3']:.1f}] {units.get(var, '')}".rstrip()
             for var in variables if (r := reference.get(var) or {}).get("median") is not None]
    if not lines:
        return ""
    return ("\n\nTypical values in this ICU cohort (all training patients, median "
            "[interquartile range]), for reference:\n" + "\n".join(lines))


def error_kind(e: Exception) -> str:
    """The type of a failed LLM call, as named in tracking.ERROR_MESSAGES."""
    if isinstance(e, requests.exceptions.Timeout):
        return "llm_timeout"
    if isinstance(e, requests.exceptions.RequestException):
        return "llm_server_error"
    if "truncated at max_tokens" in str(e):
        return "truncated_forecast"
    if "context overflow" in str(e):
        return "context_overflow"
    return "malformed_forecast"


def _naive_fallback(obs: np.ndarray, horizon_hours: int, variables: list[str],
                    fill: list[tuple[float, float]] | None = None) -> dict:
    """Last value carried forward (baselines.naive_forecast): an unobserved
    variable gets fill[i] = (training median, half-width), or 0 without it."""
    forecast, halfwidth = {}, {}
    for i, var in enumerate(variables):
        valid = obs[:, i][~np.isnan(obs[:, i])]
        if len(valid) == 0 and fill is not None:
            value, hw = fill[i]
        else:
            value = float(valid[-1]) if len(valid) else 0.0
            hw = max(float(np.std(valid)) if len(valid) > 1 else 1.0, 1e-3) * 1.5
        forecast[var] = [float(value)] * horizon_hours
        halfwidth[var] = float(hw)
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


def _halfwidth_schema(interval: str, horizon_hours: int) -> dict:
    if interval == "per_hour":
        return {"type": "array", "items": {"type": "number"}, "minItems": horizon_hours, "maxItems": horizon_hours}
    if interval == "endpoints":
        return {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
                "required": ["start", "end"]}
    return {"type": "number"}


def forecast_schema(variables: list[str], horizon_hours: int, interval: str = "constant") -> dict:
    """JSON schema for the forecast, passed to Ollama as `format` so decoding
    is constrained to exactly horizon_hours values per variable. With plain
    JSON mode, Qwen2.5-32B often ran on to 30-31 values (job 23206117: 118
    of 128 forecasts under similarity_context=trajectory fell back)."""
    return {
        "type": "object",
        "properties": {
            "forecast": {
                "type": "object",
                "properties": {var: {"type": "array", "items": {"type": "number"},
                                     "minItems": horizon_hours, "maxItems": horizon_hours}
                               for var in variables},
                "required": list(variables),
            },
            "interval_halfwidth": {
                "type": "object",
                "properties": {var: _halfwidth_schema(interval, horizon_hours) for var in variables},
                "required": list(variables),
            },
        },
        "required": ["forecast", "interval_halfwidth"],
    }


class ForecastingAgent:
    def __init__(self, llm: LocalLLM, cfg: dict, reference: dict[str, dict] | None = None):
        """`reference`: the training cohort's per-variable stats
        (harmonization_agent.reference_stats), for unobserved variables and
        cohort_anchor."""
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
        #   trend_hint, cohort_anchor, interval: see the module docstring
        fa = cfg.get("forecasting_agent") or {}
        self.similarity_context = fa.get("similarity_context", "horizon_mean")
        self.strict_length = bool(fa.get("strict_length", False))
        self.recent_hours = int(fa.get("recent_hours", 0))
        self.trend_hint = fa.get("trend_hint", "slope")
        self.cohort_anchor = bool(fa.get("cohort_anchor", False))
        self.interval = fa.get("interval", "constant")
        if self.trend_hint not in TREND_HINTS:
            raise ValueError(f"forecasting_agent.trend_hint must be one of {TREND_HINTS}, not {self.trend_hint!r}")
        if self.interval not in INTERVAL_MODES:
            raise ValueError(f"forecasting_agent.interval must be one of {INTERVAL_MODES}, not {self.interval!r}")
        if self.cohort_anchor and reference is None:
            raise ValueError("forecasting_agent.cohort_anchor needs the training cohort's reference stats")
        self.reference = reference
        self._fill = naive_fill(reference, self.variables)
        # A value outside its valid range widened by 10x the range on each
        # side is not a forecast (Med42: -2999999954): the variable is filled.
        self._sane = {var: (lo - 10 * (hi - lo), hi + 10 * (hi - lo)) for var, (lo, hi) in valid_ranges(cfg).items()}
        self._hard = hard_ranges(cfg)
        self.system_prompt = system_prompt(self.interval)
        self.schema = forecast_schema(self.variables, self.horizon_hours, self.interval)
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

    def build_prompt(self, obs: np.ndarray, similarity_context: dict | None = None) -> str:
        """The user prompt forecast() sends (src/probe_output.py sends it too)."""
        obs_summary = summarize_observation(obs, self.variables)
        # With cohort_anchor the cohort's values are listed below already.
        obs_block = _format_observation_block(obs_summary, self.units, trend=self.trend_hint != "none",
                                              reference=None if self.cohort_anchor else self.reference)

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

        anchor_block = (_format_cohort_anchor(self.reference, self.variables, self.units)
                        if self.cohort_anchor else "")
        trend_block = f"\n\n{DAMPED_TREND_NOTE}" if self.trend_hint == "damped" else ""
        prompt = (
            f"Observation window summary ({len(self.variables)} variables, most recent "
            f"reading first in 'last'):\n{obs_block}{anchor_block}{similarity_block}{trend_block}\n\n"
            f"Forecast each variable for the next {self.horizon_hours} hours, one value per "
            "hour with one decimal place, as strict JSON per the system instructions."
        )
        if self.strict_length:
            prompt += (f" Every variable listed above, including any with no observations, "
                       f"must have exactly {self.horizon_hours} values — no more, no fewer.")
        return prompt

    def forecast(self, obs: np.ndarray, similarity_context: dict | None = None) -> dict:
        prompt = self.build_prompt(obs, similarity_context)
        raw = parsed = None
        try:
            raw = self.llm.generate(prompt, system=self.system_prompt, json_mode=True, schema=self.schema)
            parsed = extract_json_block(raw)
            missing = self._validate_shape(parsed)
            missing += self._absurd_variables(parsed, missing)
            if len(missing) == len(self.variables):
                raise ValueError("no usable variable in forecast")
            halfwidths = self._halfwidths(parsed, missing)
            if missing:
                self._fill_missing(parsed, obs, missing)
            for var in self.variables:
                lo, hi = self._hard[var]
                parsed["forecast"][var] = np.clip(np.asarray(parsed["forecast"][var], dtype=float), lo, hi).tolist()
            # One half-width per hour for every variable (filled ones: constant).
            parsed["interval_halfwidth"] = {
                var: halfwidths.get(var) or [float(parsed["interval_halfwidth"][var])] * self.horizon_hours
                for var in self.variables}
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
            return _naive_fallback(obs, self.horizon_hours, self.variables, self._fill)

    def _halfwidths(self, parsed: dict, missing: list[str]) -> dict[str, list[float]]:
        """The model's interval half-widths as one value per hour, whatever
        the interval mode (a number, a list of horizon_hours values, or
        {start, end} interpolated). Raises on a shape the forecast can't use."""
        out = {}
        for var in self.variables:
            if var in missing:
                continue
            hw = parsed["interval_halfwidth"].get(var)
            if isinstance(hw, (int, float)) and not isinstance(hw, bool):
                vals = [float(hw)] * self.horizon_hours
            elif isinstance(hw, list) and len(hw) == self.horizon_hours:
                vals = [float(x) for x in hw]
            elif isinstance(hw, dict) and {"start", "end"} <= set(hw):
                vals = np.linspace(float(hw["start"]), float(hw["end"]), self.horizon_hours).tolist()
            else:
                raise ValueError(f"interval_halfwidth for '{var}' has an unusable shape: {str(hw)[:80]}")
            out[var] = [abs(x) for x in vals]
        return out

    def _absurd_variables(self, parsed: dict, missing: list[str]) -> list[str]:
        """Variables whose forecast holds a non-finite value or one far
        outside its valid range; they are filled like left-out ones and
        counted as such (n_filled), plus under error_counts."""
        out = []
        for var in self.variables:
            if var in missing:
                continue
            vals = np.asarray(parsed["forecast"][var], dtype=float)
            lo, hi = self._sane[var]
            if not np.all(np.isfinite(vals)) or np.any((vals < lo) | (vals > hi)):
                out.append(var)
        if out:
            log.warning("Forecast for %s holds non-finite or absurd values; filled instead.", ", ".join(out))
            with self._fallback_lock:
                self.error_counts["absurd_values"] = self.error_counts.get("absurd_values", 0) + len(out)
        return out

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
        naive = _naive_fallback(obs, self.horizon_hours, self.variables, self._fill)
        for var in missing:
            parsed["forecast"][var] = naive["forecast"][var]
            parsed["interval_halfwidth"][var] = naive["interval_halfwidth"][var]
        with self._fallback_lock:
            for var in missing:
                self.n_filled[var] += 1
        log.info("Forecast left out %s; filled with the naive forecast for %s.",
                 ", ".join(missing), "it" if len(missing) == 1 else "them")
