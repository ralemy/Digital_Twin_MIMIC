"""
Data-harmonization (ETL) agent — Chapter 5, Section 5.5, step 2.

Converts the long-format resampled panel (stay_id, variable, hour, value) into
fixed-shape per-patient tensors: an observation-window array and a
forecast-horizon array, both (n_hours, n_variables), with linear interpolation
of short internal gaps and explicit NaN for hours with no observation at all
(carried through to evaluation, which ignores NaN — see src/metrics.py).

This is a standalone "agent" in the architecture sense used throughout the
proposal: a narrow, single-purpose local process with a defined input/output
contract, not literally a separate service. It is called directly as a Python
function by pipeline.py.
"""
from __future__ import annotations

# from pathlib import Path
import numpy as np
import pandas as pd

from common import get_logger, valid_ranges  #, load_config

log = get_logger("harmonization_agent")

VARIABLES = None  # set at runtime from config


def _pivot_stay(panel_stay: pd.DataFrame, total_hours: int, variables: list[str]) -> np.ndarray:
    """Return a (total_hours, n_variables) array for one stay, hour-indexed 0..total_hours-1."""
    arr = np.full((total_hours, len(variables)), np.nan, dtype=float)
    for _, row in panel_stay.iterrows():
        h = int(row["hour"])
        if 0 <= h < total_hours:
            v_idx = variables.index(row["variable"])
            arr[h, v_idx] = row["value"]
    return arr


def _interpolate_short_gaps(arr: np.ndarray, max_gap_hours: int = 3) -> np.ndarray:
    """Linearly interpolate gaps up to max_gap_hours; longer gaps stay NaN."""
    out = arr.copy()
    _, n_vars = arr.shape
    for v in range(n_vars):
        series = pd.Series(out[:, v])
        interpolated = series.interpolate(method="linear", limit=max_gap_hours, limit_area="inside")
        out[:, v] = interpolated.values
    return out


def drop_invalid_values(panel_long: pd.DataFrame, cfg: dict, value_col: str = "value") -> pd.DataFrame:
    """panel_long without the values outside each variable's valid range
    (common.valid_ranges), logging how many were dropped per variable."""
    ranges = valid_ranges(cfg)
    lo = panel_long["variable"].map(lambda v: ranges.get(v, (-np.inf, np.inf))[0])
    hi = panel_long["variable"].map(lambda v: ranges.get(v, (-np.inf, np.inf))[1])
    bad = (panel_long[value_col] < lo) | (panel_long[value_col] > hi)
    if bad.any():
        counts = panel_long.loc[bad, "variable"].value_counts()
        log.info("Dropped %d value(s) outside the variables' valid ranges: %s", int(bad.sum()),
                 ", ".join(f"{var} {n} (valid {ranges[var][0]}-{ranges[var][1]})" for var, n in counts.items()))
    return panel_long[~bad]


def build_tensors(cfg: dict, cohort: pd.DataFrame, panel_long: pd.DataFrame) -> dict[int, dict]:
    """
    Returns {stay_id: {"obs": (obs_h, n_var) array, "horizon": (hor_h, n_var) array}}.
    Hourly values outside a variable's valid range are dropped first: new
    panels are filtered per raw value at extraction, but a panel extracted
    before that still holds hourly means of artefacts.
    """
    variables = [v["name"] for v in cfg["variables"]]
    obs_h = cfg["cohort"]["observation_window_hours"]
    hor_h = cfg["cohort"]["forecast_horizon_hours"]
    total_h = obs_h + hor_h
    panel_long = drop_invalid_values(panel_long, cfg)

    tensors = {}
    grouped = panel_long.groupby("stay_id")
    for stay_id in cohort["stay_id"]:
        panel_stay = grouped.get_group(stay_id) if stay_id in grouped.groups else pd.DataFrame(columns=panel_long.columns)
        full = _pivot_stay(panel_stay, total_h, variables)
        full = _interpolate_short_gaps(full)
        tensors[int(stay_id)] = {
            "obs": full[:obs_h, :],
            "horizon": full[obs_h:, :],
        }
    log.info("Built observation/horizon tensors for %d stays (%d obs hours, %d horizon hours, %d variables).",
              len(tensors), obs_h, hor_h, len(variables))
    return tensors


def summarize_observation(obs: np.ndarray, variables: list[str]) -> dict[str, dict]:
    """Compact per-variable summary of the observation window, used both as
    features for the ML baselines and as the structured input handed to the
    LLM forecasting agent's prompt."""
    summary = {}
    for i, var in enumerate(variables):
        series = obs[:, i]
        valid = series[~np.isnan(series)]
        if len(valid) == 0:
            summary[var] = {"last": None, "mean": None, "std": None, "slope": None, "n_obs": 0}
            continue
        # simple slope via least squares against hour index of valid points
        idx = np.where(~np.isnan(series))[0]
        slope = float(np.polyfit(idx, valid, 1)[0]) if len(valid) >= 2 else 0.0
        summary[var] = {
            "last": float(valid[-1]),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)) if len(valid) > 1 else 0.0,
            "slope": slope,
            "n_obs": len(valid),
        }
    return summary


def reference_stats(tensors: dict[int, dict], variables: list[str]) -> dict[str, dict]:
    """Per variable, the median, interquartile range and standard deviation of
    every value (observation and horizon hours) in `tensors` — the training
    split. Used where a patient has no value of their own: the naive forecast
    and the LLM fallback fill an unobserved variable with the median instead
    of 0, and the forecasting prompt names it as the cohort's typical value."""
    stack = np.concatenate([np.concatenate([t["obs"], t["horizon"]]) for t in tensors.values()])
    out = {}
    for i, var in enumerate(variables):
        vals = stack[:, i][~np.isnan(stack[:, i])]
        if len(vals) == 0:
            out[var] = {"median": None, "q1": None, "q3": None, "std": None}
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        out[var] = {"median": float(med), "q1": float(q1), "q3": float(q3), "std": float(np.std(vals))}
    return out


def naive_fill(reference: dict[str, dict] | None, variables: list[str]) -> list[tuple[float, float]] | None:
    """(value, ~90% half-width) per variable for a variable with no
    observations: the training median, +/- 1.645 training standard
    deviations. None without reference stats (the old fill: 0 +/- 1.5)."""
    if reference is None:
        return None
    fill = []
    for var in variables:
        r = reference.get(var) or {}
        fill.append((r["median"], 1.645 * r["std"]) if r.get("median") is not None else (0.0, 1.5))
    return fill
