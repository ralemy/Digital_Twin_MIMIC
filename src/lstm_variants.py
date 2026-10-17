"""
LSTM baseline variants re-trained on finished runs' patients and split
(LSTM-only reruns: no LLM; the GPU only speeds up training). At the full
scope (19 variables, 14 sparse labs) the runs' LSTM (lstm_target level) is
worse than naive; this fits the variants below and forecasts both the
validation patients (to choose a variant) and the test patients (to compare
the chosen one with the other conditions), so the choice never looks at the
test patients.

VARIANTS, fixed before any was trained (2026-10-16):
  level        the runs' own LSTM, re-trained (a reproduction check against
               the run's lstm_raw.npz)
  delta        lstm_target delta: predicts the change from each variable's
               last measured value (baselines.LSTMBaseline)
  delta_e400   delta trained for 400 epochs instead of the config's 200
               (the tuning grid's epochs400)
Everything else is the run's config (lstm_input ffill_mask, hidden 128,
lr 1e-3, seed 42).

For each tuned config and variant, the forecasts go to
<results_dir>/lstm_variants/<variant>/lstm_raw.npz (test patients) and
.../<variant>/val/lstm_raw.npz (validation patients), each with stay_ids,
y_true, y_pred, y_lower, y_upper, next to the run's own results. A variant
already saved is skipped. src/pool_runs.py compares a variant's test forecasts with
the other conditions through `overrides` (config/pooled_full_v3.yaml).

Usage:
    python src/lstm_variants.py <tuned config> [<tuned config> ...] [--variants level,delta,delta_e400]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from common import get_logger, load_config
from evaluate_results import per_patient_smape
from harmonization_agent import build_tensors
from pipeline import fit_models, run_condition
from run_experiment import load_cohort_and_panel, split_tensors

log = get_logger("lstm_variants")

VARIANTS = {
    "level": {"lstm_target": "level"},
    "delta": {"lstm_target": "delta"},
    "delta_e400": {"lstm_target": "delta", "lstm_epochs": 400},
}


def lstm_cfg(cfg: dict, settings: dict) -> dict:
    """The config with only the LSTM to build, with these baselines settings."""
    out = copy.deepcopy(cfg)
    out["conditions"] = ["lstm"]
    out["ensembles"] = []
    out["baselines"].update({"run_gbm": False, "run_lstm": True, "run_single_model_llm": False, **settings})
    return out


def save(path: Path, result: dict) -> float:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, stay_ids=np.array(result["stay_ids"]), y_true=result["y_true"], y_pred=result["y_pred"],
             y_lower=result["y_lower"], y_upper=result["y_upper"])
    tmp.rename(path)
    return float(np.nanmean(per_patient_smape(result["y_true"], result["y_pred"])))


def run_config(config_path: str, variants: list[str]) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg["paths"]["results_dir"]) / "lstm_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    reference = Path(cfg["paths"]["results_dir"]) / "lstm_raw.npz"

    for name in variants:
        paths = {"val": out_dir / name / "val" / "lstm_raw.npz", "test": out_dir / name / "lstm_raw.npz"}
        if all(p.exists() for p in paths.values()):
            log.info("%s: variant %s already saved — skipped.", config_path, name)
            continue
        vcfg = lstm_cfg(cfg, VARIANTS[name])
        # No checkpoint: the variant's weights are not kept, only its forecasts.
        fitted = fit_models(vcfg, splits["train"])
        scores = {}
        for split, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            result = run_condition("lstm", vcfg, splits["train"], splits[split], fitted)
            scores[split] = save(path, result)
        log.info("%s: variant %s — per-patient sMAPE val %.4f, test %.4f", config_path, name,
                 scores["val"], scores["test"])
        if name == "level" and reference.exists():
            with np.load(reference) as ref, np.load(paths["test"]) as new:
                same_ids = np.array_equal(ref["stay_ids"], new["stay_ids"])
                diff = float(np.nanmax(np.abs(ref["y_pred"] - new["y_pred"]))) if same_ids else float("nan")
            log.info("%s: level is the run's own LSTM — same patients: %s; max |difference| from its "
                     "lstm_raw.npz %.4g.", config_path, same_ids, diff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="+", help="finished runs' tuned configs")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated, of {list(VARIANTS)}")
    args = parser.parse_args()
    variants = [v for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants {sorted(unknown)}; choose from {list(VARIANTS)}")
    for config_path in args.configs:
        run_config(config_path, variants)


if __name__ == "__main__":
    main()
