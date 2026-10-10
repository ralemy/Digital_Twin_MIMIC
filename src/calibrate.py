"""
Interval calibration on the validation split (split-conformal), run after
tuning and before the final experiment's evaluation.

Every condition in the config forecasts the CALIBRATION subset of the
validation partition (the validation stays not used for tuning; see
src/tune.py). For each condition and variable, the conformity score of an
observed cell is its error relative to the forecast's own interval
half-width,
    r = |y - y_hat| / w,
and the calibration factor is the conformal quantile of the scores at the
nominal level 1 - alpha (alpha = 1 - evaluation.interval_width_pct / 100):
    f = the ceil((n + 1)(1 - alpha))-th smallest r.
Scaling a forecast's half-widths by f gives intervals with approximately the
nominal coverage. src/evaluate_results.py applies the factors to the test
forecasts and reports coverage and width before and after calibration.

Resuming: forecasts are checkpointed per batch under
<checkpoint_dir>/calibration/ and calibration.json is rewritten after each
finished condition; --full-refresh starts over.

Usage:
    python src/calibrate.py --config-file config/config_alliance_lean_tuned.yaml [--grid config/tuning_grid.yaml] [--full-refresh]
Writes:
    <results_dir>/calibration.json
    <results_dir>/calibration_smape.npz   per-patient sMAPE of each condition on
                                          the calibration patients, for
                                          src/select_rq_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tracking
from checkpoint import RunCheckpoint, atomic_path, checkpoint_root, condition_fingerprint
from critic_agent import critic_review_stage
from common import LLM_CONDITIONS, ensure_work_dirs, get_logger, load_config, parse_step_args, split_condition
from evaluate_results import per_patient_smape
from harmonization_agent import build_tensors
from metrics import interval_coverage, mean_interval_width
from common import hard_ranges
from forecasting_agent import postprocess_arrays, postprocess_params
from pipeline import ensemble_result, ensembles, fit_models, run_condition, validate_conditions
from run_experiment import load_cohort_and_panel, split_tensors
from tune import DEFAULT_GRID, _digest, config_grid, load_grid, validation_subsets

log = get_logger("calibrate")


# Post-processing parameters tried per LLM condition (fit_postprocess).
# Bounded so the model always keeps a share of the forecast: anchor weight 1
# with damping 0 would replace it by the recent_mean baseline outright (the
# end-to-end test with a mock LLM fitted exactly that), and the condition
# would no longer measure the LLM.
DAMPING_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)
ANCHOR_GRID = (0.0, 0.25, 0.5, 0.75)


def calibration_key(cfg: dict, condition: str, train_ids) -> str:
    """Identifies the condition's settings independently of which patients
    were forecast, so evaluate_results.py can check the factors were fitted
    for the same settings as the test forecasts. An ensemble's key is made of
    its members' keys."""
    members = ensembles(cfg).get(condition)
    if members:
        return _digest([calibration_key(cfg, m, train_ids) for m in members])
    fp = condition_fingerprint(cfg, condition, train_ids, [])
    fp.pop("test_ids", None)
    return _digest(fp)


def conformal_factor(y_true, y_pred, halfwidth, alpha: float) -> float | None:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(halfwidth)) & (halfwidth > 0)
    scores = np.sort(np.abs(y_true[mask] - y_pred[mask]) / halfwidth[mask])
    n = len(scores)
    if n == 0:
        return None
    rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(scores[rank - 1])


def apply_factors(y_pred, y_lower, y_upper, factors: dict, variables: list[str]):
    """Intervals scaled per variable around the forecast."""
    lo, hi = y_lower.copy(), y_upper.copy()
    for i, var in enumerate(variables):
        f = factors.get(var)
        if f is None:
            continue
        lo[..., i] = y_pred[..., i] - f * (y_pred[..., i] - y_lower[..., i])
        hi[..., i] = y_pred[..., i] + f * (y_upper[..., i] - y_pred[..., i])
    return lo, hi


def fit_postprocess(result: dict, obs: np.ndarray, cfg: dict, base: str) -> tuple[dict, dict]:
    """The post-processing parameters (drift damping, level anchor weight)
    that minimise mean per-patient sMAPE on these patients, chosen over
    DAMPING_GRID x ANCHOR_GRID from the forecasts before post-processing
    (result["y_raw"]), so no LLM call is repeated. A condition with a critic
    is scored after clipping to the plausible range, its critic's last step;
    with critic_agent.stage: raw, y_raw is already the critic's output, so
    nothing is clipped. Returns (params, {"smape_config": ..., "smape_fitted": ...})."""
    variables = [v["name"] for v in cfg["variables"]]
    hard = np.array([hard_ranges(cfg)[var] for var in variables])
    plausible = np.array([v["plausible_range"] for v in cfg["variables"]], dtype=float)
    critic = (base in ("full_pipeline", "full_pipeline_no_similarity", "full_pipeline_clip_critic")
              and critic_review_stage(cfg) == "final")
    window = postprocess_params(cfg)["level_window_hours"]

    def forecast(params):
        out = postprocess_arrays(result["y_raw"], obs, params, hard)
        return np.clip(out, plausible[:, 0], plausible[:, 1]) if critic else out

    best, best_score = None, np.inf
    for lam in DAMPING_GRID:
        for w in ANCHOR_GRID:
            params = {"drift_damping": lam, "level_anchor_weight": w, "level_window_hours": window}
            score = float(np.nanmean(per_patient_smape(result["y_true"], forecast(params))))
            if score < best_score - 1e-12:
                best, best_score = params, score
    configured = float(np.nanmean(per_patient_smape(result["y_true"], result["y_pred"])))
    return best, {"smape_config": configured, "smape_fitted": best_score, "y_pred": forecast(best)}


def main(config_path: str, grid_path: str | None = None, full_refresh: bool = False) -> None:
    cfg = load_config(config_path)
    validate_conditions(cfg)
    ensure_work_dirs(cfg)
    grid = load_grid(grid_path or config_grid(cfg))
    # A tuned config records the subset it was tuned with; use the same split.
    n_tune = (cfg.get("tuned_from") or {}).get("n_tune_patients", grid["n_tune_patients"])
    seed = (cfg.get("tuned_from") or {}).get("seed", grid["seed"])
    alpha = 1 - cfg["evaluation"]["interval_width_pct"] / 100
    variables = [v["name"] for v in cfg["variables"]]

    checkpoint = RunCheckpoint(checkpoint_root(cfg) / "calibration")
    if full_refresh:
        log.info("--full-refresh: deleting calibration checkpoints in %s.", checkpoint.root)
        checkpoint.clear()
    checkpoint.claim()

    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    _, calib_ids = validation_subsets(splits["val"], n_tune, seed)
    calib = {sid: splits["val"][sid] for sid in calib_ids}
    train_ids = sorted(splits["train"])
    log.info("Calibrating on %d validation patients (disjoint from the %d tuning patients), "
             "nominal coverage %.0f%%.", len(calib_ids), n_tune, 100 * (1 - alpha))

    tracking.start(cfg, "calibrate", config_path, {"n_calibration_patients": len(calib_ids)})
    fitted = fit_models(cfg, splits["train"], checkpoint=checkpoint)
    out_path = Path(cfg["paths"]["results_dir"]) / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"alpha": alpha, "n_calibration_patients": len(calib_ids), "config": config_path,
              "conditions": {}}
    smape_path = out_path.parent / "calibration_smape.npz"
    per_patient = {}
    results = {}                # per condition, for the ensembles
    fit_pp = bool((cfg.get("forecasting_agent") or {}).get("fit_postprocess", False))

    def record(condition: str, result: dict, extra: dict) -> None:
        """Conformal factors etc. of one condition's calibration forecasts
        into calibration.json (rewritten after each condition)."""
        halfwidth = (result["y_upper"] - result["y_lower"]) / 2
        factors = {var: conformal_factor(result["y_true"][..., i], result["y_pred"][..., i],
                                         halfwidth[..., i], alpha) for i, var in enumerate(variables)}
        lo, hi = apply_factors(result["y_pred"], result["y_lower"], result["y_upper"], factors, variables)
        pp = per_patient_smape(result["y_true"], result["y_pred"])
        output["conditions"][condition] = {
            "key": calibration_key(cfg, condition, train_ids),
            "factors": factors,
            "coverage_before": interval_coverage(result["y_true"], result["y_lower"], result["y_upper"]),
            "coverage_after": interval_coverage(result["y_true"], lo, hi),
            # Per variable, each in its own unit.
            "width_before": {var: mean_interval_width(result["y_lower"][..., i], result["y_upper"][..., i])
                             for i, var in enumerate(variables)},
            "width_after": {var: mean_interval_width(lo[..., i], hi[..., i]) for i, var in enumerate(variables)},
            "smape_mean": float(np.nanmean(pp)),
            **extra,
        }
        per_patient[condition] = pp
        log.info("Condition '%s': factors %s; calibration-set coverage %.3f -> %.3f",
                 condition, {k: (round(v, 2) if v else v) for k, v in factors.items()},
                 output["conditions"][condition]["coverage_before"],
                 output["conditions"][condition]["coverage_after"])
        checkpoint.check_owner()
        with atomic_path(out_path) as tmp:
            tmp.write_text(json.dumps(output, indent=2))
        with atomic_path(smape_path) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, stay_ids=np.array(sorted(calib_ids)), **per_patient)
        c = output["conditions"][condition]
        tracking.log_metrics({"calibrate/conditions_done": len(output["conditions"]),
                              **{f"calibrate/{condition}/{k}": c[k] for k in
                                 ("coverage_before", "coverage_after") if c[k] is not None}})

    for condition in cfg["conditions"]:
        base, variant = split_condition(condition)
        if (base in LLM_CONDITIONS and variant is None and "forecaster" not in fitted) \
                or (base in ("gbm", "lstm") and base not in fitted):
            log.warning("Skipping '%s' — model not built for this config.", condition)
            continue
        fp = condition_fingerprint(cfg, condition, train_ids, calib_ids)
        result = run_condition(condition, cfg, splits["train"], calib, fitted,
                               checkpoint=checkpoint.condition(condition, fp),
                               progress=tracking.progress_logger("calibrate/"))
        tracking.log_errors(f"calibrate/{condition}", result["llm_errors"])
        if (result.get("llm_errors") or {}).get("zero_interval"):
            log.info("Condition '%s': %d interval half-width(s) of 0 replaced (per variable and patient).",
                     condition, result["llm_errors"]["zero_interval"])
        extra = {
            "llm_fallbacks": result["llm_fallbacks"],
            "llm_fallback_rate": (result["llm_fallbacks"] / max(1, len(result["stay_ids"]))
                                  if result["llm_fallbacks"] is not None else None),
            "llm_filled": (dict(zip(variables, map(int, result["llm_filled"])))
                           if result["llm_filled"] is not None else None),
        }
        if fit_pp and base in LLM_CONDITIONS and not np.all(np.isnan(result["y_raw"])):
            # Post-processing fitted for this condition on the calibration
            # patients, from its forecasts before post-processing; the factors
            # below are then fitted to the post-processed forecasts, and
            # run_experiment.py applies the same parameters to the test ones.
            obs = np.stack([calib[sid]["obs"] for sid in result["stay_ids"]])
            params, fit = fit_postprocess(result, obs, cfg, base)
            halfwidth = (result["y_upper"] - result["y_lower"]) / 2
            result = {**result, "y_pred": fit["y_pred"], "y_lower": fit["y_pred"] - halfwidth,
                      "y_upper": fit["y_pred"] + halfwidth}
            extra["postprocess"] = params
            extra["postprocess_fit"] = {k: fit[k] for k in ("smape_config", "smape_fitted")}
            log.info("Condition '%s': post-processing fitted on the calibration patients: drift_damping %.2f, "
                     "level_anchor_weight %.2f (sMAPE %.4f with the config's settings -> %.4f)", condition,
                     params["drift_damping"], params["level_anchor_weight"], fit["smape_config"], fit["smape_fitted"])
        results[condition] = result
        record(condition, result, extra)

    for name, members in ensembles(cfg).items():
        if all(m in results for m in members):
            record(name, ensemble_result([results[m] for m in members]), {"members": members})
        else:
            log.warning("Skipping ensemble '%s' — a member was skipped.", name)

    log.info("Calibration factors written to %s — evaluate_results.py applies them to the test forecasts.", out_path)


def _add_arguments(parser) -> None:
    parser.add_argument("--grid", default=None,
                        help="tuning grid file, for the tuning/calibration split "
                             f"(default: the config's tuning_grid, else {DEFAULT_GRID})")
    parser.add_argument("--full-refresh", action="store_true",
                        help="delete the calibration checkpoints and start over (default: resume)")


if __name__ == "__main__":
    args = parse_step_args("Split-conformal interval calibration on the validation split.",
                           add_arguments=_add_arguments)
    try:
        main(args.config_file, args.grid, full_refresh=args.full_refresh)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
