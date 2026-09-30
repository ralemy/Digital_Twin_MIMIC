"""
Statistical analysis — Chapter 5, Section 5.6.

RQ1: paired Wilcoxon signed-rank test over per-patient sMAPE,
     full_pipeline vs single_model_llm (capacity-matched baseline).
RQ2: full_pipeline vs full_pipeline_no_critic, primary outcome =
     physiological-plausibility violation rate; also reports whether sMAPE
     is significantly worse (it should not be, per the control-theoretic
     prediction in the Theoretical Framework).
RQ3: stratifies the test cohort into stable / deteriorating subgroups
     (new vasopressor initiation within the horizon window) and recomputes
     the primary metrics within each subgroup.

All headline metrics are reported with bootstrap 95% confidence intervals,
resampling PATIENTS (not individual observations) with replacement, per
Section 5.6.

Usage:
    python src/evaluate_results.py --config-file config/config.yaml
Requires run_experiment.py to have been run first (reads <results_dir>/*_raw.npz).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import get_logger, load_config, parse_step_args
from metrics import smape

log = get_logger("evaluate_results")


def load_raw(results_dir: Path, condition: str) -> dict | None:
    path = results_dir / f"{condition}_raw.npz"
    if not path.exists():
        log.warning("No raw results found for condition '%s' at %s — skipping any comparison involving it.", condition, path)
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def per_patient_smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(n_patients,) array of per-patient sMAPE, pooled across variables/hours."""
    n_patients = y_true.shape[0]
    out = np.full(n_patients, np.nan)
    for i in range(n_patients):
        out[i] = smape(y_true[i], y_pred[i])
    return out


def bootstrap_ci(values: np.ndarray, n_resamples: int, alpha: float, rng: np.random.RandomState) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    point = float(np.mean(values))
    boot_means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lo, hi


def compare_conditions(name_a: str, name_b: str, data_a: dict, data_b: dict, cfg: dict) -> dict:
    """Paired comparison over patients common to both conditions."""
    ids_a, ids_b = list(data_a["stay_ids"]), list(data_b["stay_ids"])
    common = sorted(set(ids_a) & set(ids_b))
    idx_a = [ids_a.index(i) for i in common]
    idx_b = [ids_b.index(i) for i in common]

    smape_a = per_patient_smape(data_a["y_true"][idx_a], data_a["y_pred"][idx_a])
    smape_b = per_patient_smape(data_b["y_true"][idx_b], data_b["y_pred"][idx_b])

    mask = ~(np.isnan(smape_a) | np.isnan(smape_b))
    stat, pval = (np.nan, np.nan)
    if mask.sum() >= 10 and not np.allclose(smape_a[mask], smape_b[mask]):
        stat, pval = wilcoxon(smape_a[mask], smape_b[mask])

    rng = np.random.RandomState(cfg["cohort"]["random_seed"])
    mean_a, lo_a, hi_a = bootstrap_ci(smape_a, cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_alpha"], rng)
    mean_b, lo_b, hi_b = bootstrap_ci(smape_b, cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_alpha"], rng)

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "n_paired_patients": int(mask.sum()),
        f"{name_a}_smape_mean": mean_a, f"{name_a}_smape_ci95": [lo_a, hi_a],
        f"{name_b}_smape_mean": mean_b, f"{name_b}_smape_ci95": [lo_b, hi_b],
        "wilcoxon_statistic": float(stat) if not np.isnan(stat) else None, # type: ignore
        "wilcoxon_pvalue": float(pval) if not np.isnan(pval) else None, # type: ignore
    }


def plausibility_violation_comparison(name_a: str, name_b: str, data_a: dict, data_b: dict, cfg: dict) -> dict:
    from metrics import plausibility_violation_rate

    variables = [v["name"] for v in cfg["variables"]]
    pv_a = plausibility_violation_rate(data_a["y_pred"], variables)
    pv_b = plausibility_violation_rate(data_b["y_pred"], variables)
    overall_a = float(np.nanmean(list(pv_a.values())))
    overall_b = float(np.nanmean(list(pv_b.values())))
    return {
        "comparison": f"{name_a}_vs_{name_b}_plausibility",
        f"{name_a}_violation_rate": overall_a,
        f"{name_b}_violation_rate": overall_b,
        "per_variable": {"a": pv_a, "b": pv_b},
    }


def rq3_subgroup_analysis(cfg: dict, results_dir: Path) -> dict | None:
    work_dir = Path(cfg["paths"]["work_dir"])
    vaso_path = Path(cfg["paths"]["cache_dir"]) / "vasopressor_events.parquet"
    cohort_path = work_dir / "cohort.parquet"
    if not vaso_path.exists():
        log.warning("No vasopressor_events.parquet found; RQ3 subgroup analysis skipped.")
        return None

    cohort = pd.read_parquet(cohort_path)
    vaso = pd.read_parquet(vaso_path)
    obs_h = cfg["cohort"]["observation_window_hours"]
    lookahead_h = cfg["deterioration_labels"]["lookahead_hours"]

    vaso = vaso.merge(cohort[["stay_id", "intime"]], on="stay_id")
    vaso["hour"] = (pd.to_datetime(vaso["starttime"]) - pd.to_datetime(vaso["intime"])).dt.total_seconds() / 3600
    deteriorating_ids = set(
        vaso[(vaso["hour"] >= obs_h) & (vaso["hour"] <= obs_h + lookahead_h)]["stay_id"].unique()
    )
    log.info("RQ3: %d / %d test-eligible stays flagged deteriorating (new vasopressor in horizon window).",
              len(deteriorating_ids), cohort["stay_id"].nunique())

    # _ = [v["name"] for v in cfg["variables"]]
    subgroup_results = {}
    for condition in ("single_model_llm", "full_pipeline"):
        data = load_raw(results_dir, condition)
        if data is None:
            continue
        ids = list(data["stay_ids"])
        stable_idx = [i for i, sid in enumerate(ids) if sid not in deteriorating_ids]
        deteriorating_idx = [i for i, sid in enumerate(ids) if sid in deteriorating_ids]

        cond_result = {}
        for label, idx in (("stable", stable_idx), ("deteriorating", deteriorating_idx)):
            if len(idx) == 0:
                cond_result[label] = {"n": 0}
                continue
            per_patient = per_patient_smape(data["y_true"][idx], data["y_pred"][idx])
            cond_result[label] = {
                "n": len(idx),
                "smape_mean": float(np.nanmean(per_patient)),
                "smape_std": float(np.nanstd(per_patient)),
            }
        subgroup_results[condition] = cond_result

    return {"deteriorating_stay_ids": sorted(deteriorating_ids), "results_by_condition": subgroup_results}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    results_dir = Path(cfg["paths"]["results_dir"])
    output = {}

    single = load_raw(results_dir, "single_model_llm")
    full = load_raw(results_dir, "full_pipeline")
    no_critic = load_raw(results_dir, "full_pipeline_no_critic")

    if single is not None and full is not None:
        output["RQ1_orchestration_vs_single_model"] = compare_conditions(
            "single_model_llm", "full_pipeline", single, full, cfg
        )
    else:
        log.warning("Skipping RQ1 comparison — missing single_model_llm or full_pipeline results.")

    if full is not None and no_critic is not None:
        output["RQ2_critic_accuracy_tradeoff"] = compare_conditions(
            "full_pipeline_no_critic", "full_pipeline", no_critic, full, cfg
        )
        output["RQ2_critic_plausibility"] = plausibility_violation_comparison(
            "full_pipeline_no_critic", "full_pipeline", no_critic, full, cfg
        )
    else:
        log.warning("Skipping RQ2 comparison — missing full_pipeline or full_pipeline_no_critic results.")

    rq3 = rq3_subgroup_analysis(cfg, results_dir)
    if rq3 is not None:
        output["RQ3_deterioration_subgroup"] = rq3

    out_path = results_dir / "statistical_analysis.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    log.info("Wrote statistical analysis to %s", out_path)

    for key, val in output.items():
        log.info("--- %s ---\n%s", key, json.dumps(val, indent=2, default=str)[:2000])


if __name__ == "__main__":
    doc = __doc__ or "Statistical analysis — Chapter 5, Section 5.6."
    args = parse_step_args(doc.strip().splitlines()[0])
    main(args.config_file)
