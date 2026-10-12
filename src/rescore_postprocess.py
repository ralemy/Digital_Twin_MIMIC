"""
Post-processing refitted with another strategy and applied to a finished
run's saved forecasts, without new LLM calls (no GPU; light enough for a
login node). The run's own results are not touched: the rescored forecasts
go to a separate directory.

calibrate.py fits one drift damping and one level-anchor weight per LLM
condition, shared by every variable, over DAMPING_GRID x ANCHOR_GRID, and
run_experiment.py applies them. With 14 sparse laboratory values in the full
panel the best settings differ between vital signs and labs. --strategy:
  global        one setting for all variables (calibrate.py's fit; with
                --grid standard it must reproduce calibration.json, a check)
  vitals_labs   one setting for the vital signs, one for the laboratory values
                (variables with source hosp_labevents)
  per_variable  one setting per variable
--grid standard is calibrate.py's grid; extended adds damping 0 and 0.05 and
anchor weight 1.0.

For each LLM condition with saved pre-post-processing forecasts (y_raw):
the settings are fitted on the calibration patients (mean per-patient
sMAPE, as calibrate.py), applied to the test patients' y_raw, and the
interval factors refitted on the re-post-processed calibration forecasts
(split-conformal, as calibrate.py). Written to <out-dir> (default
<results_dir>/rescored_<strategy>_<grid>/): <condition>_raw.npz in the run's
format (with the model's own half-widths around the new forecast; the
factors are in rescore.json), the config's ensembles whose members changed,
and rescore.json (settings, factors, calibration and test sMAPE and coverage
before and after). src/pool_runs.py can compare them through `overrides`.

--cv prints a two-fold cross-validation on the calibration patients only
(fit on one half, score the other) for every strategy, and writes nothing.

Usage:
    python src/rescore_postprocess.py <tuned config> --strategy vitals_labs --grid extended [--conditions a,b] [--out-dir DIR]
    python src/rescore_postprocess.py <tuned config> --cv
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from calibrate import ANCHOR_GRID, DAMPING_GRID, apply_factors, conformal_factor
from common import LLM_CONDITIONS, split_condition
from evaluate_results import per_patient_smape
from metrics import interval_coverage
from offline_runs import FinishedRun
from pipeline import ensemble_result, ensembles

GRIDS = {"standard": (DAMPING_GRID, ANCHOR_GRID),
         "extended": (tuple(sorted(set(DAMPING_GRID) | {0.0, 0.05})), tuple(sorted(set(ANCHOR_GRID) | {1.0})))}
STRATEGIES = ("global", "vitals_labs", "per_variable")


def contributions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(n_vars,) terms whose sum is the mean per-patient sMAPE (each patient's
    sMAPE is the mean over its observed cells), so the objective splits by
    variable and groups of variables can be fitted separately."""
    s = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    s = np.where(np.isnan(y_true) | np.isnan(y_pred), np.nan, s)
    n = np.sum(~np.isnan(s), axis=(1, 2))
    keep = n > 0
    return (np.nansum(s[keep], axis=1) / n[keep, None]).mean(axis=0)


def groups(run: FinishedRun, strategy: str) -> list[np.ndarray]:
    nv = len(run.variables)
    if strategy == "global":
        return [np.ones(nv, dtype=bool)]
    if strategy == "vitals_labs":
        return [g for g in (~run.is_lab, run.is_lab) if g.any()]
    return [np.arange(nv) == i for i in range(nv)]


def fit(run: FinishedRun, condition: str, y_raw, obs, y_true, strategy: str, grid: str, table=None):
    """(lam, w) per variable, each a (n_vars,) array, minimising mean
    per-patient sMAPE; ties keep calibrate.py's order (first best)."""
    damping, anchor = GRIDS[grid]
    if table is None:
        table = {p: contributions(y_true, run.postprocess(condition, y_raw, obs, *p)) for p in product(damping, anchor)}
    lam, w = np.zeros(len(run.variables)), np.zeros(len(run.variables))
    for g in groups(run, strategy):
        best, score = None, np.inf
        for p in product(damping, anchor):
            v = table[p][g].sum()
            if v < score - 1e-12:
                best, score = p, v
        lam[g], w[g] = best
    return lam, w


def cross_validate(run: FinishedRun, condition: str) -> dict:
    cal = run.calibration_forecasts(condition)
    folds = [np.arange(len(cal["stay_ids"])) % 2 == k for k in (0, 1)]
    out = {}
    for grid in GRIDS:
        damping, anchor = GRIDS[grid]
        tables = [{p: contributions(cal["y_true"][f], run.postprocess(condition, cal["y_raw"][f], cal["obs"][f], *p))
                   for p in product(damping, anchor)} for f in folds]
        for strategy in STRATEGIES:
            total = 0.0
            for k in (0, 1):
                lam, w = fit(run, condition, None, None, None, strategy, grid, table=tables[k])
                ev = folds[1 - k]
                total += contributions(cal["y_true"][ev], run.postprocess(condition, cal["y_raw"][ev], cal["obs"][ev], lam, w)).sum()
            out[f"{strategy}/{grid}"] = total / 2
    return out


def rescore(run: FinishedRun, condition: str, strategy: str, grid: str, alpha: float) -> tuple[dict, dict]:
    cal, test = run.calibration_forecasts(condition), run.test_forecasts(condition)
    lam, w = fit(run, condition, cal["y_raw"], cal["obs"], cal["y_true"], strategy, grid)
    cal_half = (cal["y_upper"] - cal["y_lower"]) / 2
    cal_pred = run.postprocess(condition, cal["y_raw"], cal["obs"], lam, w)
    factors = {v: conformal_factor(cal["y_true"][..., i], cal_pred[..., i], cal_half[..., i], alpha)
               for i, v in enumerate(run.variables)}
    half = (test["y_upper"] - test["y_lower"]) / 2
    pred = run.postprocess(condition, test["y_raw"], test["obs"], lam, w)
    new = {**test, "y_pred": pred, "y_lower": pred - half, "y_upper": pred + half}
    new.pop("obs")
    old_lo, old_hi = apply_factors(test["y_pred"], test["y_lower"], test["y_upper"], run.factors(condition), run.variables)
    new_lo, new_hi = apply_factors(pred, new["y_lower"], new["y_upper"], factors, run.variables)
    pp = (run.calibration["conditions"].get(condition) or {}).get("postprocess") or {}
    report = {
        "settings": {v: {"drift_damping": float(lam[i]), "level_anchor_weight": float(w[i])} for i, v in enumerate(run.variables)},
        "run_settings": pp,
        "factors": factors,
        "calibration_smape": {"run": float(np.nanmean(per_patient_smape(cal["y_true"], cal["y_pred"]))),
                              "rescored": float(np.nanmean(per_patient_smape(cal["y_true"], cal_pred)))},
        "test_smape": {"run": float(np.nanmean(per_patient_smape(test["y_true"], test["y_pred"]))),
                       "rescored": float(np.nanmean(per_patient_smape(test["y_true"], pred)))},
        "test_coverage_calibrated": {"run": interval_coverage(test["y_true"], old_lo, old_hi),
                                     "rescored": interval_coverage(test["y_true"], new_lo, new_hi)},
    }
    if strategy == "global" and pp:
        report["reproduces_run_settings"] = bool(np.allclose(lam, pp["drift_damping"]) and np.allclose(w, pp["level_anchor_weight"]))
    return new, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the finished run's tuned config")
    parser.add_argument("--strategy", choices=STRATEGIES, default="global")
    parser.add_argument("--grid", choices=tuple(GRIDS), default="standard")
    parser.add_argument("--conditions", default=None, help="comma-separated LLM conditions (default: all with saved y_raw)")
    parser.add_argument("--out-dir", default=None, help="default <results_dir>/rescored_<strategy>_<grid>")
    parser.add_argument("--cv", action="store_true", help="only print a two-fold CV of every strategy on the calibration patients")
    args = parser.parse_args()

    run = FinishedRun(args.config)
    alpha = 1 - run.cfg["evaluation"]["interval_width_pct"] / 100
    llm = [c for c in run.conditions() if split_condition(c)[0] in LLM_CONDITIONS]
    conditions = args.conditions.split(",") if args.conditions else llm

    if args.cv:
        keys = [f"{s}/{g}" for g in GRIDS for s in STRATEGIES]
        print("Two-fold CV on the calibration patients, mean per-patient sMAPE; columns: change from global/standard.")
        print(f"{'condition':42s} {'global/standard':>15s} " + " ".join(f"{k:>22s}" for k in keys[1:]))
        for c in conditions:
            r = cross_validate(run, c)
            print(f"{c:42s} {r['global/standard']:15.4f} " + " ".join(f"{r[k] - r['global/standard']:+22.4f}" for k in keys[1:]))
        return

    out_dir = Path(args.out_dir) if args.out_dir else run.results_dir / f"rescored_{args.strategy}_{args.grid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports, rescored = {}, {}
    for c in conditions:
        new, rep = rescore(run, c, args.strategy, args.grid, alpha)
        np.savez_compressed(out_dir / f"{c}_raw.npz", **new)
        reports[c], rescored[c] = rep, new
        print(f"{c:42s} test sMAPE {rep['test_smape']['run']:.4f} -> {rep['test_smape']['rescored']:.4f}  "
              f"coverage {rep['test_coverage_calibrated']['run']:.3f} -> {rep['test_coverage_calibrated']['rescored']:.3f}"
              + (f"  reproduces run settings: {rep['reproduces_run_settings']}" if "reproduces_run_settings" in rep else ""))
    for name, members in ensembles(run.cfg).items():
        if not any(m in rescored for m in members):
            continue
        parts = [rescored.get(m) or run.test_forecasts(m) for m in members]
        e = ensemble_result([{k: p[k] for k in ("y_true", "y_pred", "y_lower", "y_upper", "stay_ids")} for p in parts])
        np.savez_compressed(out_dir / f"{name}_raw.npz", **{k: np.asarray(v) for k, v in e.items()
                                                            if k in ("y_true", "y_pred", "y_lower", "y_upper", "stay_ids")})
        print(f"{name:42s} rebuilt from {' + '.join(members)}")
    (out_dir / "rescore.json").write_text(json.dumps(
        {"config": args.config, "strategy": args.strategy, "grid": args.grid, "conditions": reports}, indent=2, default=float))
    print("Written to", out_dir)


if __name__ == "__main__":
    main()
