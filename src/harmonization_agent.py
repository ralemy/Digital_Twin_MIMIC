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

from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, get_logger

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
    n_hours, n_vars = arr.shape
    for v in range(n_vars):
        series = pd.Series(out[:, v])
        interpolated = series.interpolate(method="linear", limit=max_gap_hours, limit_area="inside")
        out[:, v] = interpolated.values
    return out


def build_tensors(cfg: dict, cohort: pd.DataFrame, panel_long: pd.DataFrame) -> dict[int, dict]:
    """
    Returns {stay_id: {"obs": (obs_h, n_var) array, "horizon": (hor_h, n_var) array}}.
    """
    variables = [v["name"] for v in cfg["variables"]]
    obs_h = cfg["cohort"]["observation_window_hours"]
    hor_h = cfg["cohort"]["forecast_horizon_hours"]
    total_h = obs_h + hor_h

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
            "n_obs": int(len(valid)),
        }
    return summary
