"""
A finished run's saved forecasts, for offline analyses that need no LLM
calls and no GPU (src/diagnose_coverage.py, src/rescore_postprocess.py).

For one condition it gives the calibration patients' forecasts (from
checkpoints_tuned/calibration/conditions/<condition>/) and the test
patients' (from <results_dir>/<condition>_raw.npz), each with the truth, the
observation window and, for LLM conditions, the forecast before
post-processing (y_raw). Nothing here writes to the run.

An LLM condition's calibration forecasts are re-post-processed with the
parameters calibrate.py fitted (calibration.json), exactly as calibrate.py
did before it fitted the interval factors, so they match what the factors
were fitted on. Conditions with a critic that reviews the post-processed
forecast (critic_agent.stage: final) are clipped to the plausible range
afterwards, the stand-in calibrate.py uses for that critic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import LLM_CONDITIONS, hard_ranges, load_config, split_condition
from critic_agent import critic_review_stage
from forecasting_agent import postprocess_arrays
from harmonization_agent import build_tensors
from run_experiment import load_cohort_and_panel, split_tensors

CRITIC_BASES = ("full_pipeline", "full_pipeline_no_similarity", "full_pipeline_clip_critic")


class FinishedRun:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.variables = [v["name"] for v in self.cfg["variables"]]
        self.results_dir = Path(self.cfg["paths"]["results_dir"])
        self.checkpoint_dir = Path(self.cfg["paths"]["checkpoint_dir"]) if self.cfg["paths"].get("checkpoint_dir") \
            else Path(self.cfg["paths"]["work_dir"]) / "checkpoints"
        self.hard = np.array([hard_ranges(self.cfg)[v] for v in self.variables])
        self.plausible = np.array([v["plausible_range"] for v in self.cfg["variables"]], dtype=float)
        self.is_lab = np.array([v.get("source") == "hosp_labevents" for v in self.cfg["variables"]])
        self.window = (self.cfg.get("forecasting_agent") or {}).get("level_window_hours", 6)
        cal_path = self.results_dir / "calibration.json"
        self.calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {"conditions": {}}
        cohort, panel = load_cohort_and_panel(self.cfg)
        splits = split_tensors(build_tensors(self.cfg, cohort, panel), cohort)
        self.tensors = {**splits["val"], **splits["test"]}

    def conditions(self) -> list[str]:
        """Conditions with both calibration forecasts and test results."""
        root = self.checkpoint_dir / "calibration" / "conditions"
        return [c for c in self.cfg["conditions"]
                if (root / c).is_dir() and (self.results_dir / f"{c}_raw.npz").exists()]

    def clips(self, condition: str) -> bool:
        return split_condition(condition)[0] in CRITIC_BASES and critic_review_stage(self.cfg) == "final"

    def postprocess(self, condition: str, y_raw: np.ndarray, obs: np.ndarray, lam, w) -> np.ndarray:
        """y_raw post-processed with drift damping `lam` and anchor weight `w`,
        each a scalar or a (n_vars,) array (one value per variable)."""
        lam = np.broadcast_to(np.asarray(lam, dtype=float), (len(self.variables),))
        w = np.broadcast_to(np.asarray(w, dtype=float), (len(self.variables),))
        out = np.empty_like(y_raw)
        for pair in {(float(a), float(b)) for a, b in zip(lam, w)}:
            cols = (lam == pair[0]) & (w == pair[1])
            params = {"drift_damping": pair[0], "level_anchor_weight": pair[1], "level_window_hours": self.window}
            out[..., cols] = postprocess_arrays(y_raw, obs, params, self.hard)[..., cols]
        if self.clips(condition):
            out = np.clip(out, self.plausible[:, 0], self.plausible[:, 1])
        return out

    def calibration_forecasts(self, condition: str) -> dict:
        """The calibration patients' forecasts as calibrate.py fitted the
        interval factors on them."""
        files = sorted((self.checkpoint_dir / "calibration" / "conditions" / condition).glob("batch_*.npz"))
        parts = [dict(np.load(f)) for f in files]
        ids = np.concatenate([p["stay_ids"] for p in parts])
        out = {"stay_ids": ids,
               "y_pred": np.concatenate([p["y_pred"] for p in parts]),
               "y_lower": np.concatenate([p["y_lower"] for p in parts]),
               "y_upper": np.concatenate([p["y_upper"] for p in parts]),
               "y_raw": np.concatenate([p["y_raw"] for p in parts]) if "y_raw" in parts[0] else None}
        out["y_true"] = np.stack([self.tensors[int(s)]["horizon"] for s in ids])
        out["obs"] = np.stack([self.tensors[int(s)]["obs"] for s in ids])
        pp = (self.calibration["conditions"].get(condition) or {}).get("postprocess")
        if pp and out["y_raw"] is not None and split_condition(condition)[0] in LLM_CONDITIONS:
            half = (out["y_upper"] - out["y_lower"]) / 2
            out["y_pred"] = self.postprocess(condition, out["y_raw"], out["obs"],
                                             pp["drift_damping"], pp["level_anchor_weight"])
            out["y_lower"], out["y_upper"] = out["y_pred"] - half, out["y_pred"] + half
        return out

    def test_forecasts(self, condition: str) -> dict:
        with np.load(self.results_dir / f"{condition}_raw.npz") as z:
            out = {k: z[k] for k in z.files}
        out["obs"] = np.stack([self.tensors[int(s)]["obs"] for s in out["stay_ids"]])
        if "y_raw" not in out:
            # Runs from before *_raw.npz carried y_raw (e.g. tri_lean_exp3.4):
            # the step 3 checkpoints keep it per batch.
            files = sorted((self.checkpoint_dir / "run_experiment" / "conditions" / condition).glob("batch_*.npz"))
            parts = [dict(np.load(f)) for f in files]
            if parts and all("y_raw" in p for p in parts):
                ids = np.concatenate([p["stay_ids"] for p in parts])
                if np.array_equal(ids, out["stay_ids"]):
                    out["y_raw"] = np.concatenate([p["y_raw"] for p in parts])
        return out

    def factors(self, condition: str) -> dict:
        return (self.calibration["conditions"].get(condition) or {}).get("factors") or {}
