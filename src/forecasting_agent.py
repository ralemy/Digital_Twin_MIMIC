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


def _naive_fallback(obs: np.ndarray, horizon_hours: int, variables: list[str]) -> dict:
    forecast, halfwidth = {}, {}
    for i, var in enumerate(variables):
        valid = obs[:, i][~np.isnan(obs[:, i])]
        last = float(valid[-1]) if len(valid) else 0.0
        std = float(np.std(valid)) if len(valid) > 1 else 1.0
        forecast[var] = [last] * horizon_hours
        halfwidth[var] = max(std, 1e-3) * 1.5
    return {"forecast": forecast, "interval_halfwidth": halfwidth}


class ForecastingAgent:
    def __init__(self, llm: LocalLLM, cfg: dict):
        self.llm = llm
        self.variables = [v["name"] for v in cfg["variables"]]
        self.units = {v["name"]: v["unit"] for v in cfg["variables"]}
        self.horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
        # How many forecast() calls fell back to the naive forecast. Reported
        # per condition: a model that often fails to produce valid JSON would
        # otherwise just look like the naive baseline. Locked because
        # pipeline.py calls forecast() from a thread pool.
        self.n_fallbacks = 0
        self._fallback_lock = threading.Lock()

    def forecast(self, obs: np.ndarray, similarity_context: dict | None = None) -> dict:
        obs_summary = summarize_observation(obs, self.variables)
        obs_block = _format_observation_block(obs_summary, self.units)

        similarity_block = ""
        if similarity_context is not None:
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

        try:
            raw = self.llm.generate(prompt, system=SYSTEM_PROMPT, json_mode=True)
            parsed = extract_json_block(raw)
            self._validate_shape(parsed)
            return parsed
        except requests.exceptions.ConnectionError:
            # The Ollama server is gone (job ending, crash): not a model
            # failure, so stop the run rather than record a naive forecast.
            raise
        except Exception as e:  # noqa: BLE001 - deliberately broad: any parse/shape failure falls back
            log.warning("Forecasting agent LLM call failed or malformed (%s); using naive fallback.", e)
            with self._fallback_lock:
                self.n_fallbacks += 1
            return _naive_fallback(obs, self.horizon_hours, self.variables)

    def _validate_shape(self, parsed: dict) -> None:
        if "forecast" not in parsed or "interval_halfwidth" not in parsed:
            raise ValueError("missing top-level keys")
        for var in self.variables:
            if var not in parsed["forecast"]:
                raise ValueError(f"missing variable '{var}' in forecast")
            vals = parsed["forecast"][var]
            if not isinstance(vals, list) or len(vals) != self.horizon_hours:
                raise ValueError(f"variable '{var}' forecast has wrong length: {len(vals) if isinstance(vals, list) else type(vals)}")
