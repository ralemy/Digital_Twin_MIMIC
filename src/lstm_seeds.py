"""
The LSTM baseline re-trained with other random seeds, on a finished run's
patients and split, to see how much the comparisons against it depend on the
one seed every run trains it with (baselines.lstm_seed, 42). No LLM; the GPU
only speeds up training (a few minutes per seed).

For each config (a run's tuned config, e.g.
tuned_configs/tri_lean_exp4/config_alliance_lean_exp4_tuned.yaml) and each
seed, the LSTM is fitted on the run's training patients exactly as
pipeline.fit_models fits it (same settings, only lstm_seed changed) and
forecasts its test patients through pipeline.run_condition. The forecasts go
to <results_dir>/lstm_seeds/lstm_seed<N>_raw.npz (stay_ids, y_true, y_pred,
y_lower, y_upper), next to the run's own results, where jobs/pack_run.sh packs
them. A seed already saved is skipped.

The run's own seed is re-trained too, as a check: its forecasts are compared
with the run's lstm_raw.npz (GPU training need not be bit-for-bit
reproducible, so the difference is reported, not required to be zero).

src/pool_runs.py then compares the pipeline and the ensemble with each seed's
LSTM (config/pooled_lean_lstm_seeds.yaml).

Usage:
    python src/lstm_seeds.py --seeds 42,43,44,45,46 <tuned config> [<tuned config> ...]
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

log = get_logger("lstm_seeds")


def lstm_cfg(cfg: dict, seed: int) -> dict:
    """The config with only the LSTM to build, trained with this seed."""
    out = copy.deepcopy(cfg)
    out["conditions"] = ["lstm"]
    out["ensembles"] = []
    out["baselines"].update({"run_gbm": False, "run_lstm": True, "run_single_model_llm": False, "lstm_seed": seed})
    return out


def run_config(config_path: str, seeds: list[int]) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg["paths"]["results_dir"]) / "lstm_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    own_seed = cfg["baselines"].get("lstm_seed")
    reference = Path(cfg["paths"]["results_dir"]) / "lstm_raw.npz"

    for seed in seeds:
        path = out_dir / f"lstm_seed{seed}_raw.npz"
        if path.exists():
            log.info("%s: seed %d already saved (%s) — skipped.", config_path, seed, path)
            continue
        scfg = lstm_cfg(cfg, seed)
        fitted = fit_models(scfg, splits["train"])
        result = run_condition("lstm", scfg, splits["train"], splits["test"], fitted)
        smape = float(np.nanmean(per_patient_smape(result["y_true"], result["y_pred"])))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, stay_ids=np.array(result["stay_ids"]), y_true=result["y_true"], y_pred=result["y_pred"],
                 y_lower=result["y_lower"], y_upper=result["y_upper"])
        tmp.rename(path)
        log.info("%s: seed %d — test per-patient sMAPE %.4f, saved to %s", config_path, seed, smape, path)
        if seed == own_seed and reference.exists():
            with np.load(reference) as ref:
                same_ids = np.array_equal(ref["stay_ids"], np.array(result["stay_ids"]))
                diff = float(np.nanmax(np.abs(ref["y_pred"] - result["y_pred"]))) if same_ids else float("nan")
                ref_smape = float(np.nanmean(per_patient_smape(ref["y_true"], ref["y_pred"])))
            log.info("%s: seed %d is the run's own — same patients: %s; max |difference| from its lstm_raw.npz "
                     "%.4g; sMAPE %.4f here vs %.4f in the run.", config_path, seed, same_ids, diff, smape, ref_smape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="+", help="finished runs' tuned configs")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="comma-separated LSTM seeds (default 42-46)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for config_path in args.configs:
        run_config(config_path, seeds)


if __name__ == "__main__":
    main()
