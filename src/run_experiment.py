"""
Main entry point — Chapter 5, Section 5.5, steps 2-4.

Loads the extracted cohort/panel (src/extract_cohort.py output), builds
observation/horizon tensors, fits every trainable baseline once, runs every
configured condition over the held-out test split, and writes raw forecast
arrays + a per-condition metrics summary to <results_dir>.

Usage:
    python src/run_experiment.py --config-file config/config.yaml
    python src/run_experiment.py --config-file config/config.yaml --full-refresh

Resuming: progress is checkpointed under <work_dir>/checkpoints/run_experiment/
(see src/checkpoint.py) — GBM per variable, LSTM every few epochs, and every
condition per batch of performance.checkpoint_batch_size test patients. If a
run is stopped (e.g. a Slurm time limit), running the same command again
continues where it left off; finished conditions are not re-run. Each run
rewrites all_conditions_summary.csv after every finished condition.
--full-refresh deletes the checkpoints first and starts from scratch — use it
after changing prompts or code, which checkpoints can't detect (config or
cohort changes are detected and stop the run with a message).

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

from checkpoint import RunCheckpoint, condition_fingerprint
from common import (
    LLM_CONDITIONS,
    cfg_for_llm_variant,
    ensure_work_dirs,
    get_logger,
    load_config,
    ollama_model_name,
    parse_step_args,
    split_condition,
)
from harmonization_agent import build_tensors
from metrics import evaluate_twin
from pipeline import fit_models, run_condition, validate_conditions

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


def main(config_path: str, full_refresh: bool = False) -> None:
    cfg = load_config(config_path)
    ensure_work_dirs(cfg)
    validate_conditions(cfg)

    checkpoint = RunCheckpoint(Path(cfg["paths"]["work_dir"]) / "checkpoints" / "run_experiment")
    if full_refresh:
        log.info("--full-refresh: deleting checkpoints in %s and starting from scratch.", checkpoint.root)
        checkpoint.clear()
    elif checkpoint.exists():
        log.info("Resuming from checkpoints in %s (use --full-refresh to start over).", checkpoint.root)
    # From here on this run owns the checkpoints; an older run still writing
    # to them (e.g. a cancelled job that hasn't fully stopped) is refused.
    checkpoint.claim()

    cohort, panel_long = load_cohort_and_panel(cfg)
    tensors = build_tensors(cfg, cohort, panel_long)
    splits = split_tensors(tensors, cohort)

    fitted_models = fit_models(cfg, splits["train"], checkpoint=checkpoint)
    train_ids, test_ids = sorted(splits["train"]), sorted(splits["test"])

    results_dir = Path(cfg["paths"]["results_dir"])
    variables = [v["name"] for v in cfg["variables"]]
    summary_rows = []

    for condition in cfg["conditions"]:
        base, variant = split_condition(condition)
        needs_llm = base in LLM_CONDITIONS
        if needs_llm and variant is None and "forecaster" not in fitted_models:
            log.warning("Skipping condition '%s' — local LLM not configured/available.", condition)
            continue
        if base == "gbm" and "gbm" not in fitted_models:
            log.warning("Skipping condition '%s' — GBM baseline disabled in config.", condition)
            continue
        if base == "lstm" and "lstm" not in fitted_models:
            log.warning("Skipping condition '%s' — LSTM baseline disabled in config.", condition)
            continue

        # The Ollama name (alias if set) the condition ran against, recorded
        # with its results so each row says which model produced it.
        llm_model = ollama_model_name(cfg_for_llm_variant(cfg, variant)["llm"]) if needs_llm else None
        cond_checkpoint = checkpoint.condition(
            condition, condition_fingerprint(cfg, condition, train_ids, test_ids))
        batch_size = max(1, int(cfg.get("performance", {}).get("checkpoint_batch_size", 32)))
        done_rows = cond_checkpoint.load_rows(n_batches=-(-len(test_ids) // batch_size))
        if done_rows is not None:
            log.info("Condition '%s' already finished in an earlier run — results loaded from checkpoint.", condition)
            summary_rows.extend(done_rows)
            continue

        if llm_model:
            log.info("Condition '%s' uses LLM '%s'.", condition, llm_model)

        result = run_condition(condition, cfg, splits["train"], splits["test"], fitted_models,
                               checkpoint=cond_checkpoint)

        # Share of test patients whose LLM forecast failed (no answer, bad
        # JSON, wrong shape) and was replaced by the naive forecast. Compare
        # it across models before comparing their accuracy.
        fallback_rate = None
        if result["llm_fallbacks"] is not None:
            fallback_rate = result["llm_fallbacks"] / max(1, len(result["stay_ids"]))
            log_fn = log.warning if fallback_rate > 0 else log.info
            log_fn("Condition '%s': %d / %d LLM forecasts fell back to naive (%.1f%%).",
                   condition, result["llm_fallbacks"], len(result["stay_ids"]), 100 * fallback_rate)

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

        condition_rows = []
        for var, pm in report.point_metrics.items():
            condition_rows.append({
                "condition": condition, "llm_model": llm_model, "variable": var,
                "smape": pm["smape"], "mae": pm["mae"], "rmse": pm.get("rmse"),
                "ks_statistic": report.ks_results.get(var, {}).get("ks_statistic"),
                "plausibility_violation_rate": report.plausibility_violations.get(var),
                "interval_coverage": report.coverage,
                "mean_interval_width": report.mean_width,
                "llm_fallback_rate": fallback_rate,
            })

        # Saving the rows marks the condition finished: a resumed run skips it.
        cond_checkpoint.save_rows(condition_rows)
        summary_rows.extend(condition_rows)
        _write_summary(summary_rows, results_dir)

    _write_summary(summary_rows, results_dir)
    log.info("All conditions complete. Combined summary: %s", results_dir / "all_conditions_summary.csv")
    log.info("Next step: python src/evaluate_results.py --config-file %s", config_path)


def _write_summary(summary_rows: list[dict], results_dir: Path) -> None:
    """(Re)write all_conditions_summary.csv with every condition finished so
    far, in `conditions` order — so partial results survive a stopped run."""
    pd.DataFrame(summary_rows).to_csv(results_dir / "all_conditions_summary.csv", index=False)


def _add_arguments(parser) -> None:
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="delete all checkpoints of this config's run and start from scratch "
             "(default: resume from them)",
    )


if __name__ == "__main__":
    doc = __doc__ or "Main entry point — Chapter 5, Section 5.5, steps 2-4."
    args = parse_step_args(doc.strip().splitlines()[0], add_arguments=_add_arguments)
    main(args.config_file, full_refresh=args.full_refresh)
