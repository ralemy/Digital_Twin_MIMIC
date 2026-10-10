"""
Critic / validation agent — Chapter 5's architecture, theoretically grounded in
Wiener's (1948) feedback-loop concept (Theoretical Framework, Section 4.1):
the forecasting agent's output is the observed state, the plausible range is
the reference standard, and a bounded correction call is the corrective
action. Per the proposal's control-theoretic framing, this is expected to
reduce the physiological-plausibility violation rate largely independently of
point-accuracy — it is deliberately NOT another attempt to make the forecast
"more accurate" in a general sense, only to bring implausible values back
inside clinically defensible bounds.

Runs entirely locally. The "correction call" goes to the forecaster's own
local LLM unless the LLM variant sets `critic_variant`, which gives the
critic another local model (e.g. a MedGemma forecaster with a Gemma 3
critic). With clip_only (condition full_pipeline_clip_critic) there is no
LLM call at all: out-of-range values are clipped to the plausible range,
the baseline that an LLM critic has to beat on accuracy.
"""
from __future__ import annotations

import threading

import numpy as np
import requests

from common import get_logger
from llm_client import LocalLLM, extract_json_block

log = get_logger("critic_agent")

CRITIC_STAGES = ("final", "raw")


def critic_review_stage(cfg: dict) -> str:
    """critic_agent.stage: which forecast the critic reviews.
      final (default): the post-processed forecast, i.e. what is reported.
                       The level anchor and damping keep it close to the
                       observed values, so the critic rarely has anything
                       to fix (tri_lean_exp3.4: 0 violations in every
                       condition).
      raw:             the LLM's own output, before post-processing, which
                       is then applied to the reviewed forecast."""
    stage = (cfg.get("critic_agent") or {}).get("stage", "final")
    if stage not in CRITIC_STAGES:
        raise ValueError(f"critic_agent.stage must be one of {CRITIC_STAGES}, not {stage!r}")
    return stage


CORRECTION_SYSTEM_PROMPT = (
    "You are a clinical plausibility checker for research forecasts. You will "
    "be shown a forecast for one physiological variable that contains values "
    "outside the clinically plausible range for a living ICU patient. Revise "
    "ONLY the implausible hourly values to the nearest clinically defensible "
    "value, leaving already-plausible hours unchanged. Respond with STRICT "
    'JSON only: {"corrected_values": [<hour_1>, <hour_2>, ...]}, with the same '
    "number of values as you were given."
)


class CriticAgent:
    def __init__(self, llm: LocalLLM | None, cfg: dict, enabled: bool = True, clip_only: bool = False):
        self.llm = llm
        self.enabled = enabled
        self.clip_only = clip_only
        self.variables = [v["name"] for v in cfg["variables"]]
        self.plausible_range = {v["name"]: tuple(v["plausible_range"]) for v in cfg["variables"]}
        self.units = {v["name"]: v["unit"] for v in cfg["variables"]}
        self.max_attempts = 0 if clip_only else cfg["critic_agent"]["max_correction_attempts"]
        # Failed correction calls (then clipped), for the live metrics.
        self.error_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def review(self, forecast: dict) -> dict:
        """Returns a new forecast dict plus the number of out-of-range values
        before correction (violations_before_correction) and how many of them
        the LLM's correction didn't fix, so they were clipped (clipped_values;
        all of them for a clip-only critic). The final forecast is always in
        range, so the post-critic violation rate is 0 by construction: these
        two counts are the critic's real RQ2 evidence."""
        if not self.enabled:
            n_violations = self._count_violations(forecast)
            return {"forecast": forecast["forecast"], "interval_halfwidth": forecast["interval_halfwidth"],
                    "violations_before_correction": n_violations, "clipped_values": 0, "corrected": False}

        corrected = {k: list(v) for k, v in forecast["forecast"].items()}
        n_violations_before = self._count_violations(forecast)
        n_clipped = 0
        any_correction = False

        for var, values in forecast["forecast"].items():
            lo, hi = self.plausible_range[var]
            values_arr = np.array(values, dtype=float)
            bad_idx = np.where((values_arr < lo) | (values_arr > hi))[0]
            if len(bad_idx) == 0:
                continue

            attempt = 0
            fixed = values_arr.copy()
            while len(bad_idx) > 0 and attempt < self.max_attempts:
                attempt += 1
                fixed = self._request_correction(var, fixed.tolist(), lo, hi)
                bad_idx = np.where((fixed < lo) | (fixed > hi))[0]

            if len(bad_idx) > 0:
                # Bounded fallback: clip anything the LLM still couldn't fix,
                # so a stubborn model never lets an implausible value through.
                # (A clip-only critic always ends up here, quietly.)
                fixed = np.clip(fixed, lo, hi)
                n_clipped += len(bad_idx)
                if not self.clip_only:
                    log.info("Clipped %d residual out-of-range value(s) for '%s' after %d correction attempt(s).",
                             len(bad_idx), var, attempt)

            corrected[var] = fixed.tolist()
            any_correction = True

        return {
            "forecast": corrected,
            "interval_halfwidth": forecast["interval_halfwidth"],
            "violations_before_correction": n_violations_before,
            "clipped_values": n_clipped,
            "corrected": any_correction,
        }

    def _count_violations(self, forecast: dict) -> int:
        total = 0
        for var, values in forecast["forecast"].items():
            lo, hi = self.plausible_range[var]
            values_arr = np.array(values, dtype=float)
            total += int(np.sum((values_arr < lo) | (values_arr > hi)))
        return total

    def _request_correction(self, var: str, values: list[float], lo: float, hi: float) -> np.ndarray:
        prompt = (
            f"Variable: {var} ({self.units.get(var, '')}). Clinically plausible range: "
            f"[{lo}, {hi}]. Hourly forecast values: {values}. "
            "Return corrected_values as JSON."
        )
        try:
            schema = {"type": "object", "required": ["corrected_values"],
                      "properties": {"corrected_values": {"type": "array", "items": {"type": "number"},
                                                          "minItems": len(values), "maxItems": len(values)}}}
            raw = self.llm.generate(prompt, system=CORRECTION_SYSTEM_PROMPT, json_mode=True, schema=schema)
            parsed = extract_json_block(raw)
            fixed = np.array(parsed["corrected_values"], dtype=float)
            if len(fixed) != len(values):
                raise ValueError("corrected_values length mismatch")
            return fixed
        except requests.exceptions.ConnectionError:
            raise                       # server gone: stop the run, don't clip and carry on
        except Exception as e:  # noqa: BLE001
            log.warning("Critic correction call failed for '%s' (%s); will clip instead.", var, e)
            with self._lock:
                self.error_counts["critic_correction_failed"] = self.error_counts.get("critic_correction_failed", 0) + 1
            return np.array(values, dtype=float)
