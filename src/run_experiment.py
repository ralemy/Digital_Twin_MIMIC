"""
Main entry point — Chapter 5, Section 5.5, steps 2-4.

Loads the extracted cohort/panel (src/extract_cohort.py output), builds
observation/horizon tensors, fits every trainable baseline once, runs every
configured condition over the held-out test split, and writes raw forecast
arrays + a per-condition metrics summary to <results_dir>.

Usage:
    python src/run_experiment.py --config-file config/config.yaml

Prerequisites:
    1. python src/resolve_items.py --config-file config/config.yaml
    2. python src/extract_cohort.py --config-file config/config.yaml
    3. Ollama running locally with the configured model pulled
       (skip step 3 and set baselines.run_single_model_llm: false and drop
       full_pipeline* from `conditions` in config.yaml if you only want to
       smoke-test the naive/GBM/LSTM baselines without a local LLM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_work_dirs, get_logger, load_config, parse_step_args
from harmonization_agent import build_tensors
from metrics import evaluate_twin
from pipeline import fit_models, run_condition

log = get_logger("run_experiment")


def load_cohort_and_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_dir = Path(cfg["paths"]["work_dir"])
    cohort_path, panel_path = work_dir / "cohort.parquet", work_dir / "panel_long.parquet"
    if not cohort_path.exists() or not panel_path.exists():
        raise FileNotFoundError(
            f"{cohort_path} or {panel_path} not found — run src/extract_cohort.py first."
        )
    return pd.read_parquet(cohort_path), pd.read_parquet(panel_path)


def split_tensors(tensors: dict[int, dict], cohort: pd.DataFrame) -> dict[str, dict[int, dict]]:
    split_map = cohort.set_index("stay_id")["split"].to_dict()
    out = {"train": {}, "val": {}, "test": {}}
    for sid, t in tensors.items():
        split = split_map.get(sid)
        if split in out:
            out[split][sid] = t
    for split, d in out.items():
        log.info("%s split: %d patients", split, len(d))
    return out


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_work_dirs(cfg)

    cohort, panel_long = load_cohort_and_panel(cfg)
    tensors = build_tensors(cfg, cohort, panel_long)
    splits = split_tensors(tensors, cohort)

    fitted_models = fit_models(cfg, splits["train"])

    results_dir = Path(cfg["paths"]["results_dir"])
    variables = [v["name"] for v in cfg["variables"]]
    summary_rows = []

    for condition in cfg["conditions"]:
        needs_llm = condition == "single_model_llm" or "full_pipeline" in condition
        if needs_llm and "forecaster" not in fitted_models:
            log.warning("Skipping condition '%s' — local LLM not configured/available.", condition)
            continue
        if condition == "gbm" and "gbm" not in fitted_models:
            log.warning("Skipping condition '%s' — GBM baseline disabled in config.", condition)
            continue
        if condition == "lstm" and "lstm" not in fitted_models:
            log.warning("Skipping condition '%s' — LSTM baseline disabled in config.", condition)
            continue

        result = run_condition(condition, cfg, splits["train"], splits["test"], fitted_models)

        np.savez_compressed(
            results_dir / f"{condition}_raw.npz",
            y_true=result["y_true"], y_pred=result["y_pred"],
            y_lower=result["y_lower"], y_upper=result["y_upper"],
            stay_ids=np.array(result["stay_ids"]),
        )

        report = evaluate_twin(result["y_true"], result["y_pred"], variables,
                                y_lower=result["y_lower"], y_upper=result["y_upper"])

        (results_dir / f"{condition}_summary.txt").write_text(report.summary_table())
        log.info("Condition '%s' complete:\n%s", condition, report.summary_table())

        for var, pm in report.point_metrics.items():
            summary_rows.append({
                "condition": condition, "variable": var,
                "smape": pm["smape"], "mae": pm["mae"], "rmse": pm.get("rmse"),
                "ks_statistic": report.ks_results.get(var, {}).get("ks_statistic"),
                "plausibility_violation_rate": report.plausibility_violations.get(var),
                "interval_coverage": report.coverage,
                "mean_interval_width": report.mean_width,
            })

    pd.DataFrame(summary_rows).to_csv(results_dir / "all_conditions_summary.csv", index=False)
    log.info("All conditions complete. Combined summary: %s", results_dir / "all_conditions_summary.csv")
    log.info("Next step: python src/evaluate_results.py --config-file %s", config_path)


if __name__ == "__main__":
    doc = __doc__ or "Main entry point — Chapter 5, Section 5.5, steps 2-4."
    args = parse_step_args(doc.strip().splitlines()[0])
    main(args.config_file)
