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
Model comparisons (exploratory, only when `conditions` uses LLM variants,
     e.g. full_pipeline@medgemma): for each LLM condition run with more than
     one model, every pair of models is compared on the same patients with
     the same paired sMAPE test and the plausibility violation rate.

All headline metrics are reported with bootstrap 95% confidence intervals,
resampling PATIENTS (not individual observations) with replacement, per
Section 5.6.

Usage:
    python src/evaluate_results.py --config-file config/config_local.yaml
Requires run_experiment.py to have been run first (reads <results_dir>/*_raw.npz).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import LLM_CONDITIONS, get_logger, load_config, parse_step_args, split_condition, vasopressor_cache_path
import tracking
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


def critic_record(data: dict) -> dict | None:
    """What a condition's critic did (run_experiment.py saves it with a
    full_pipeline* condition's raw results): the share of forecast values out
    of range before the critic, and the share of those its LLM corrected
    rather than it clipping them. The post-critic violation rate is 0 by
    construction (the critic ends by clipping), so these are the evidence
    for RQ2; None for results saved before they were recorded."""
    if "precritic_violations" not in data:
        return None
    n, clipped = int(data["precritic_violations"]), int(data["critic_clipped"])
    return {"precritic_violation_rate": n / max(1, int(data["n_cells"])),
            "precritic_violations": n, "clipped": clipped,
            "llm_fixed_share": (n - clipped) / n if n else None}


def plausibility_violation_comparison(name_a: str, name_b: str, data_a: dict, data_b: dict, cfg: dict) -> dict:
    from metrics import plausibility_violation_rate

    variables = [v["name"] for v in cfg["variables"]]
    ranges = {v["name"]: tuple(v["plausible_range"]) for v in cfg["variables"]}
    pv_a = plausibility_violation_rate(data_a["y_pred"], variables, ranges)
    pv_b = plausibility_violation_rate(data_b["y_pred"], variables, ranges)
    overall_a = float(np.nanmean(list(pv_a.values())))
    overall_b = float(np.nanmean(list(pv_b.values())))
    out = {
        "comparison": f"{name_a}_vs_{name_b}_plausibility",
        f"{name_a}_violation_rate": overall_a,
        f"{name_b}_violation_rate": overall_b,
        "per_variable": {"a": pv_a, "b": pv_b},
    }
    for name, data in ((name_a, data_a), (name_b, data_b)):
        record = critic_record(data)
        if record is not None:
            out[f"{name}_critic"] = record
        if "y_raw" in data:
            out[f"{name}_before_postprocess"] = before_postprocess_violations(data, variables, ranges)
    return out


def before_postprocess_violations(data: dict, variables: list[str], ranges: dict) -> dict:
    """Plausibility violations in the forecast post-processing started from
    (y_raw, saved by run_experiment.py for LLM conditions): the LLM's own
    output, or with critic_agent.stage: raw its output after the critic.
    The critic reviews the post-processed forecast by default, and the level
    anchor and damping keep that close to the observed values
    (tri_lean_exp3.4: 0 violations in every condition), so this shows what
    the model itself produces."""
    from metrics import plausibility_violation_rate

    y_raw = data["y_raw"]
    lo = np.array([ranges[v][0] for v in variables])
    hi = np.array([ranges[v][1] for v in variables])
    finite = np.isfinite(y_raw)
    n_out = int(np.sum(finite & ((y_raw < lo) | (y_raw > hi))))
    return {"violation_rate": n_out / max(1, int(finite.sum())), "violations": n_out,
            "per_variable": plausibility_violation_rate(y_raw, variables, ranges)}


def model_variant_comparisons(cfg: dict, results_dir: Path) -> list[dict]:
    """Pairwise model comparisons within each LLM condition, e.g.
    full_pipeline@gemma3 vs full_pipeline@medgemma, in `conditions` order."""
    by_base: dict[str, list[str]] = {}
    for condition in cfg["conditions"]:
        base, _ = split_condition(condition)
        if base in LLM_CONDITIONS:
            by_base.setdefault(base, []).append(condition)

    out = []
    for base, conditions in by_base.items():
        if len(conditions) < 2:
            continue
        loaded = {c: load_raw(results_dir, c) for c in conditions}
        available = [c for c in conditions if loaded[c] is not None]
        for i, name_a in enumerate(available):
            for name_b in available[i + 1:]:
                result = compare_conditions(name_a, name_b, loaded[name_a], loaded[name_b], cfg)
                result["plausibility"] = plausibility_violation_comparison(
                    name_a, name_b, loaded[name_a], loaded[name_b], cfg)
                out.append(result)
    return out


def extra_comparisons(cfg: dict, results_dir: Path) -> list[dict]:
    """The condition pairs listed under evaluation.extra_comparisons, e.g.
    which critic a MedGemma forecaster should have:
        - [full_pipeline_clip_critic@medgemma, full_pipeline@medgemma]
    Each [A, B] pair gets the same paired test as the RQs (A - B) plus the
    plausibility comparison; pairs with missing results are skipped."""
    out = []
    for pair in (cfg.get("evaluation") or {}).get("extra_comparisons") or []:
        name_a, name_b = pair
        data_a, data_b = load_raw(results_dir, name_a), load_raw(results_dir, name_b)
        if data_a is None or data_b is None:
            log.warning("Skipping extra comparison %s vs %s — results missing.", name_a, name_b)
            continue
        result = compare_conditions(name_a, name_b, data_a, data_b, cfg)
        result["plausibility"] = plausibility_violation_comparison(name_a, name_b, data_a, data_b, cfg)
        out.append(result)
    return out


def rq_analyses(cfg: dict, results_dir: Path, variant: str | None) -> dict:
    """RQ1 (single model vs full pipeline), RQ2 (critic ablation) and RQ3
    (stable vs deteriorating) for one model: llm.model (variant None) or an
    LLM variant, i.e. on its '@<variant>' conditions. RQ2 needs
    full_pipeline_no_critic@<variant> in `conditions`; a missing condition
    is skipped with a warning naming it."""
    sfx = f"@{variant}" if variant else ""
    names = {k: k + sfx for k in ("single_model_llm", "full_pipeline", "full_pipeline_no_critic")}
    single, full, no_critic = (load_raw(results_dir, names[k]) for k in names)
    out = {}
    if single is not None and full is not None:
        out["RQ1_orchestration_vs_single_model"] = compare_conditions(
            names["single_model_llm"], names["full_pipeline"], single, full, cfg)
    else:
        log.warning("Skipping RQ1%s — missing %s or %s results.", sfx, names["single_model_llm"], names["full_pipeline"])
    if full is not None and no_critic is not None:
        out["RQ2_critic_accuracy_tradeoff"] = compare_conditions(
            names["full_pipeline_no_critic"], names["full_pipeline"], no_critic, full, cfg)
        out["RQ2_critic_plausibility"] = plausibility_violation_comparison(
            names["full_pipeline_no_critic"], names["full_pipeline"], no_critic, full, cfg)
    else:
        log.warning("Skipping RQ2%s — missing %s results (add it to `conditions`).", sfx,
                    names["full_pipeline_no_critic"] if no_critic is None else names["full_pipeline"])
    rq3 = rq3_subgroup_analysis(cfg, results_dir, (names["single_model_llm"], names["full_pipeline"]))
    if rq3 is not None:
        out["RQ3_deterioration_subgroup"] = rq3
    return out


def rq3_subgroup_analysis(cfg: dict, results_dir: Path,
                          conditions=("single_model_llm", "full_pipeline")) -> dict | None:
    work_dir = Path(cfg["paths"]["work_dir"])
    vaso_path = vasopressor_cache_path(cfg)
    cohort_path = work_dir / "cohort.parquet"
    if not vaso_path.exists():
        log.warning("No %s found; RQ3 subgroup analysis skipped.", vaso_path.name)
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
    for condition in conditions:
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


def interval_calibration(cfg: dict, results_dir: Path) -> dict | None:
    """Test-set interval coverage and width per condition, before and after
    applying the split-conformal factors from <results_dir>/calibration.json
    (src/calibrate.py). Factors fitted for different settings than the test
    forecasts (key mismatch) are not applied."""
    from calibrate import apply_factors, calibration_key
    from metrics import interval_coverage, mean_interval_width

    path = results_dir / "calibration.json"
    if not path.exists():
        log.info("No %s — interval calibration not reported (run src/calibrate.py first).", path.name)
        return None
    cal = json.loads(path.read_text())
    cohort = pd.read_parquet(Path(cfg["paths"]["work_dir"]) / "cohort.parquet")
    train_ids = sorted(int(i) for i in cohort.loc[cohort["split"] == "train", "stay_id"])
    variables = [v["name"] for v in cfg["variables"]]
    out = {"alpha": cal["alpha"], "n_calibration_patients": cal["n_calibration_patients"], "conditions": {}}
    for condition, entry in cal["conditions"].items():
        data = load_raw(results_dir, condition)
        if data is None:
            continue
        if entry["key"] != calibration_key(cfg, condition, train_ids):
            log.warning("Calibration factors for '%s' were fitted with different settings — not applied.", condition)
            continue
        lo, hi = apply_factors(data["y_pred"], data["y_lower"], data["y_upper"], entry["factors"], variables)
        per_var = {}
        for i, var in enumerate(variables):
            yt, yl, yu = data["y_true"][..., i], data["y_lower"][..., i], data["y_upper"][..., i]
            per_var[var] = {"factor": entry["factors"].get(var),
                            "coverage_before": interval_coverage(yt, yl, yu),
                            "coverage_after": interval_coverage(yt, lo[..., i], hi[..., i]),
                            "width_before": mean_interval_width(yl, yu),
                            "width_after": mean_interval_width(lo[..., i], hi[..., i])}
        out["conditions"][condition] = {
            # Coverage pools all variables; widths are per variable only
            # (each in its own unit).
            "coverage_before": interval_coverage(data["y_true"], data["y_lower"], data["y_upper"]),
            "coverage_after": interval_coverage(data["y_true"], lo, hi),
            "per_variable": per_var,
        }
    return out


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    results_dir = Path(cfg["paths"]["results_dir"])
    output = {}

    # RQ1-RQ3 for each model in evaluation.rq_models: "primary" (llm.model,
    # the pre-specified analysis, reported under the plain RQ keys) and any
    # LLM variant, e.g. one chosen on validation data by
    # src/select_rq_model.py (reported under "RQ_model@<variant>").
    for model in (cfg.get("evaluation") or {}).get("rq_models") or ["primary"]:
        variant = None if model in (None, "primary") else model
        rq = rq_analyses(cfg, results_dir, variant)
        if variant is None:
            output.update(rq)
        elif rq:
            output[f"RQ_model@{variant}"] = rq

    model_comparisons = model_variant_comparisons(cfg, results_dir)
    if model_comparisons:
        output["model_variant_comparisons"] = model_comparisons

    extra = extra_comparisons(cfg, results_dir)
    if extra:
        output["extra_comparisons"] = extra

    calibration = interval_calibration(cfg, results_dir)
    if calibration:
        output["interval_calibration"] = calibration

    out_path = results_dir / "statistical_analysis.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    tracking.start(cfg, "evaluate", config_path)
    tracking.log_metrics(_aggregate_numbers(output, "evaluate"))
    log.info("Wrote statistical analysis to %s", out_path)

    for key, val in output.items():
        log.info("--- %s ---\n%s", key, json.dumps(val, indent=2, default=str)[:2000])


def _aggregate_numbers(obj, prefix: str) -> dict:
    """The numbers in the analysis (means, CIs, p-values, counts) flattened
    for the live dashboard. Anything keyed by an id (e.g. RQ3's list of
    deteriorating stay ids) is left out, so nothing identifies a patient."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if {"id", "ids"} & set(str(k).lower().split("_")):
                continue
            out.update(_aggregate_numbers(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        named = [x for x in obj if isinstance(x, dict) and "comparison" in x]
        if named:
            for x in named:
                out.update(_aggregate_numbers(x, f"{prefix}/{x['comparison']}"))
        elif prefix.endswith("ci95") and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            out[f"{prefix}_lo"], out[f"{prefix}_hi"] = obj        # a confidence interval; other lists are dropped
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


if __name__ == "__main__":
    doc = __doc__ or "Statistical analysis — Chapter 5, Section 5.6."
    args = parse_step_args(doc.strip().splitlines()[0])
    try:
        main(args.config_file)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
