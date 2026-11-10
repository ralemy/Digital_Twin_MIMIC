"""
Why a variable's calibrated intervals under-cover on the test patients
(e.g. lactate, 0.81-0.83 against 0.90 for the persistence baselines in
ror_lean_fullcohort), from a finished run's saved forecasts. No LLM, no GPU;
light enough for a login node. Reads the run, writes nothing to it.

For each condition and variable it reports:
  - coverage with the run's own factor (calibration.json): on the
    calibration patients it was fitted on and on the test patients, with a
    patient-bootstrap 90 % range for the test value (is 0.90 plausible?);
  - how many patients and cells the factor rests on, and the bootstrap
    90 % range of the factor itself (calibration patients resampled);
  - the nonconformity scores |error| / half-width in both sets (median,
    90th percentile, KS test): a shift between calibration and test;
  - test coverage split by whether the variable was measured in the
    patient's observation window, and by forecast hour (1-8, 9-16, 17-24);
  - an alternative, Mondrian conformal calibration: one factor per group
    of patients, measured vs not measured in the observation window, fitted
    on the calibration patients and applied to the test patients (opt-in,
    --mondrian; calibrate.py and evaluate_results.py are unchanged);
  - another, one factor per forecast-hour band (1-8, 9-16, 17-24), for
    errors that grow with lead time faster than the model's own half-widths
    (opt-in, --by-hour).

Usage:
    python src/diagnose_coverage.py <tuned config> [--variables lactate] [--conditions a,b] [--mondrian] [--by-hour] [--json out.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import ks_2samp

from calibrate import conformal_factor
from offline_runs import FinishedRun

HOUR_BANDS = ((0, 8), (8, 16), (16, 24))


def _cover(yt, yp, half, f):
    m = ~(np.isnan(yt) | np.isnan(yp) | np.isnan(half))
    return float(np.mean(np.abs(yt[m] - yp[m]) <= f * half[m])) if m.any() else None


def _scores(yt, yp, half):
    m = ~(np.isnan(yt) | np.isnan(yp) | np.isnan(half)) & (half > 0)
    return np.abs(yt[m] - yp[m]) / half[m]


def _patient_cover(yt, yp, half, f):
    """Per patient: (covered cells, observed cells), for bootstraps."""
    m = ~(np.isnan(yt) | np.isnan(yp) | np.isnan(half))
    hit = (np.abs(yt - yp) <= f * half) & m
    return hit.sum(axis=1), m.sum(axis=1)


def diagnose(run: FinishedRun, condition: str, var: str, alpha: float, mondrian: bool, rng, by_hour: bool = False) -> dict:
    i = run.variables.index(var)
    cal, test = run.calibration_forecasts(condition), run.test_forecasts(condition)
    f = run.factors(condition).get(var)
    if f is None:
        return {"note": "no factor in calibration.json"}
    sets = {}
    for name, d in (("calibration", cal), ("test", test)):
        yt, yp = d["y_true"][..., i], d["y_pred"][..., i]
        half = (d["y_upper"][..., i] - d["y_lower"][..., i]) / 2
        measured = ~np.all(np.isnan(d["obs"][..., i]), axis=1)
        sets[name] = (yt, yp, half, measured)

    out = {"factor": f}
    yt, yp, half, measured = sets["calibration"]
    has = ~np.all(np.isnan(yt), axis=1)
    out["calibration"] = {"coverage": _cover(yt, yp, half, f), "patients_with_values": int(has.sum()),
                          "cells": int(np.sum(~np.isnan(yt)))}
    idx = np.where(has)[0]
    boot = [conformal_factor(yt[s], yp[s], half[s], alpha) for s in
            (rng.choice(idx, size=len(idx), replace=True) for _ in range(500))]
    boot = [b for b in boot if b is not None]
    out["factor_bootstrap_90"] = [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))] if boot else None

    tyt, typ, thalf, tmeasured = sets["test"]
    hit, n = _patient_cover(tyt, typ, thalf, f)
    keep = np.where(n > 0)[0]
    bc = [hit[s].sum() / n[s].sum() for s in (rng.choice(keep, size=len(keep), replace=True) for _ in range(1000))]
    out["test"] = {"coverage": _cover(tyt, typ, thalf, f), "coverage_bootstrap_90": [float(np.percentile(bc, 5)), float(np.percentile(bc, 95))],
                   "patients_with_values": int(len(keep)), "cells": int(n.sum())}
    sc, st = _scores(yt, yp, half), _scores(tyt, typ, thalf)
    out["scores"] = {"calibration_median": float(np.median(sc)), "calibration_p90": float(np.percentile(sc, 90)),
                     "test_median": float(np.median(st)), "test_p90": float(np.percentile(st, 90)),
                     "ks_pvalue": float(ks_2samp(sc, st).pvalue)}
    out["test_by_measured_in_window"] = {
        lab: {"patients": int(np.sum(g & (n > 0))), "coverage": _cover(tyt[g], typ[g], thalf[g], f)}
        for lab, g in (("measured", tmeasured), ("not_measured", ~tmeasured))}
    out["test_by_hour"] = {f"{a + 1}-{b}": _cover(tyt[:, a:b], typ[:, a:b], thalf[:, a:b], f) for a, b in HOUR_BANDS}

    if mondrian:
        mf = {}
        for lab, g_cal in (("measured", measured), ("not_measured", ~measured)):
            mf[lab] = conformal_factor(yt[g_cal], yp[g_cal], half[g_cal], alpha) if g_cal.any() else None
        lo_t = np.where(tmeasured[:, None], mf["measured"] or f, mf["not_measured"] or f)
        res = {"factors": mf, "test_coverage": _cover(tyt, typ, thalf * lo_t, 1.0),
               "test_by_group": {lab: _cover(tyt[g], typ[g], thalf[g] * lo_t[g], 1.0)
                                 for lab, g in (("measured", tmeasured), ("not_measured", ~tmeasured))}}
        m = ~np.isnan(thalf)
        res["mean_width_ratio_vs_single_factor"] = float(np.mean((thalf * lo_t)[m]) / np.mean((thalf * f)[m]))
        out["mondrian_measured_in_window"] = res
    if by_hour:
        scale = np.ones_like(thalf)
        hf = {}
        for a, b in HOUR_BANDS:
            k = conformal_factor(yt[:, a:b], yp[:, a:b], half[:, a:b], alpha)
            hf[f"{a + 1}-{b}"] = k
            scale[:, a:b] = k if k is not None else f
        res = {"factors": hf, "test_coverage": _cover(tyt, typ, thalf * scale, 1.0),
               "test_by_hour": {f"{a + 1}-{b}": _cover(tyt[:, a:b], typ[:, a:b], (thalf * scale)[:, a:b], 1.0) for a, b in HOUR_BANDS}}
        m = ~np.isnan(thalf)
        res["mean_width_ratio_vs_single_factor"] = float(np.mean((thalf * scale)[m]) / np.mean((thalf * f)[m]))
        out["by_hour_band"] = res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the finished run's tuned config")
    parser.add_argument("--variables", default="lactate", help="comma-separated (default lactate)")
    parser.add_argument("--conditions", default=None, help="comma-separated (default: every condition with calibration and test forecasts)")
    parser.add_argument("--mondrian", action="store_true", help="also try one factor per measured/not-measured group")
    parser.add_argument("--by-hour", action="store_true", help="also try one factor per forecast-hour band")
    parser.add_argument("--json", default=None, help="also write the results to this JSON file")
    args = parser.parse_args()

    run = FinishedRun(args.config)
    alpha = 1 - run.cfg["evaluation"]["interval_width_pct"] / 100
    conditions = args.conditions.split(",") if args.conditions else run.conditions()
    rng = np.random.RandomState(42)
    out = {}
    for c in conditions:
        for var in args.variables.split(","):
            r = diagnose(run, c, var, alpha, args.mondrian, rng, by_hour=args.by_hour)
            out[f"{c}/{var}"] = r
            if "test" not in r:
                print(f"{c}/{var}: {r.get('note')}"); continue
            t, cal, s = r["test"], r["calibration"], r["scores"]
            fb = r["factor_bootstrap_90"]
            line = (f"{c:34s} {var:10s} factor {r['factor']:.2f} [{fb[0]:.2f}, {fb[1]:.2f}]  cal {cal['coverage']:.3f} "
                    f"({cal['patients_with_values']} pts)  test {t['coverage']:.3f} [{t['coverage_bootstrap_90'][0]:.3f}, "
                    f"{t['coverage_bootstrap_90'][1]:.3f}] ({t['patients_with_values']} pts)  score p90 cal {s['calibration_p90']:.2f} "
                    f"test {s['test_p90']:.2f} (KS p={s['ks_pvalue']:.2g})")
            g = r["test_by_measured_in_window"]
            line += (f"\n{'':45s}test by window: measured {g['measured']['coverage'] or float('nan'):.3f} ({g['measured']['patients']} pts), "
                     f"not measured {g['not_measured']['coverage'] or float('nan'):.3f} ({g['not_measured']['patients']} pts); by hour "
                     + ", ".join(f"{k}: {v:.3f}" for k, v in r["test_by_hour"].items() if v is not None))
            if "mondrian_measured_in_window" in r:
                m = r["mondrian_measured_in_window"]
                line += (f"\n{'':45s}Mondrian: factors {m['factors']}, test coverage {m['test_coverage']:.3f} "
                         f"(measured {m['test_by_group']['measured'] or float('nan'):.3f}, not measured "
                         f"{m['test_by_group']['not_measured'] or float('nan'):.3f}), mean width x{m['mean_width_ratio_vs_single_factor']:.2f}")
            if "by_hour_band" in r:
                h = r["by_hour_band"]
                line += (f"\n{'':45s}by hour band: factors " + ", ".join(f"{k} {v:.2f}" for k, v in h["factors"].items() if v is not None)
                         + f"; test coverage {h['test_coverage']:.3f} (" + ", ".join(f"{k}: {v:.3f}" for k, v in h["test_by_hour"].items() if v is not None)
                         + f"), mean width x{h['mean_width_ratio_vs_single_factor']:.2f}")
            print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("Written to", args.json)


if __name__ == "__main__":
    main()
