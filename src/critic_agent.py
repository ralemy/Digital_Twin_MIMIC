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

Runs entirely locally: the "correction call" is a second call to the same
local LLM, not a different or larger model.
"""
from __future__ import annotations

import numpy as np

from common import get_logger
from llm_client import LocalLLM, extract_json_block

log = get_logger("critic_agent")

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
    def __init__(self, llm: LocalLLM, cfg: dict, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled
        self.variables = [v["name"] for v in cfg["variables"]]
        self.plausible_range = {v["name"]: tuple(v["plausible_range"]) for v in cfg["variables"]}
        self.units = {v["name"]: v["unit"] for v in cfg["variables"]}
        self.max_attempts = cfg["critic_agent"]["max_correction_attempts"]

    def review(self, forecast: dict) -> dict:
        """Returns a new forecast dict plus a per-variable violation count
        (pre-correction) used for the plausibility-violation-rate metric."""
        if not self.enabled:
            n_violations = self._count_violations(forecast)
            return {"forecast": forecast["forecast"], "interval_halfwidth": forecast["interval_halfwidth"],
                    "violations_before_correction": n_violations, "corrected": False}

        corrected = {k: list(v) for k, v in forecast["forecast"].items()}
        n_violations_before = self._count_violations(forecast)
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
                fixed = np.clip(fixed, lo, hi)
                log.info("Clipped %d residual out-of-range value(s) for '%s' after %d correction attempt(s).",
                          len(bad_idx), var, attempt)

            corrected[var] = fixed.tolist()
            any_correction = True

        return {
            "forecast": corrected,
            "interval_halfwidth": forecast["interval_halfwidth"],
            "violations_before_correction": n_violations_before,
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
            raw = self.llm.generate(prompt, system=CORRECTION_SYSTEM_PROMPT, json_mode=True)
            parsed = extract_json_block(raw)
            fixed = np.array(parsed["corrected_values"], dtype=float)
            if len(fixed) != len(values):
                raise ValueError("corrected_values length mismatch")
            return fixed
        except Exception as e:  # noqa: BLE001
            log.warning("Critic correction call failed for '%s' (%s); will clip instead.", var, e)
            return np.array(values, dtype=float)
