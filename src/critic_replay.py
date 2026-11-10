"""
RQ2 with the critic before post-processing, replayed on a finished run's
saved forecasts. The run's LLM forecasts are not regenerated: every arm starts
from the same pre-post-processing forecasts (y_raw), so the arms differ only
in what the critic does.

A run with critic_agent.stage final (the default; all full_v3 runs) saves the
forecaster's own output as y_raw, before post-processing and before the
critic. That is exactly what a critic with stage raw reviews. For one LLM
condition (default full_pipeline) this script builds, on the calibration and
the test patients alike:
  no_critic        y_raw as it is
  clip_raw         y_raw clipped to the plausible ranges (clip-only critic)
  llm_critic_raw   y_raw reviewed by the condition's critic (critic_agent.py:
                   one correction request per out-of-range variable, up to
                   max_correction_attempts, then clipping), the only arm
                   that needs the LLM
then, as run_experiment.py and calibrate.py do with stage raw, post-processes
the reviewed forecast with settings fitted on the calibration patients and
refits the split-conformal interval factors. Fitting: --fit global/standard
(calibrate.py's, as pre-specified) and vitals_labs/extended
(rescore_postprocess.py's sensitivity analysis), both by default.

The run's own condition (critic after post-processing) is the fourth arm,
critic_final: as run for global/standard, and from
<results_dir>/rescored_vitals_labs_extended/ for vitals_labs/extended when
that exists. A check replays it from y_raw (run's settings, then the clipping
calibrate.py stands in for that critic) and must reproduce the run's test
forecasts.

The critic's corrections are cached in <out-dir>/llm_critic_<condition>.npz,
so re-running (e.g. with other --fit options) needs no LLM: only the first run
needs a GPU and an Ollama server (jobs/critic_replay.sh). --offline-only
skips the LLM arm (no GPU; light enough for a login node). The run's own
results are not touched.

Output in <out-dir> (default <results_dir>/critic_replay/): per arm and fit
<arm>_<strategy>_<grid>_raw.npz in the run's format (y_raw = the reviewed
forecast; src/pool_runs.py can read them through `overrides`), and
critic_replay_<condition>.json (critic counts, settings, factors, sMAPE with
bootstrap CIs, coverage, plausibility, paired comparisons).

Usage:
    python src/critic_replay.py <tuned config> [--condition full_pipeline] [--fit global/standard,vitals_labs/extended]
                                [--out-dir DIR] [--offline-only]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from calibrate import apply_factors, conformal_factor
from common import cfg_for_llm_variant, critic_variant, get_logger, split_condition
from critic_agent import CriticAgent
from evaluate_results import bootstrap_ci, per_patient_smape
from metrics import interval_coverage
from offline_runs import FinishedRun
from rescore_postprocess import GRIDS, STRATEGIES, fit

log = get_logger("critic_replay")

ARMS = ("no_critic", "clip_raw", "llm_critic_raw")
COMPARISONS = (("llm_critic_raw", "no_critic"), ("clip_raw", "no_critic"), ("llm_critic_raw", "clip_raw"),
               ("llm_critic_raw", "critic_final"), ("no_critic", "critic_final"), ("clip_raw", "critic_final"))


def violations(run: FinishedRun, y: np.ndarray) -> np.ndarray:
    """(n_vars,) values outside the plausible ranges."""
    lo, hi = run.plausible[:, 0], run.plausible[:, 1]
    return np.sum((y < lo) | (y > hi), axis=(0, 1))


def llm_review(run: FinishedRun, condition: str, y_raw: np.ndarray, workers: int) -> tuple[np.ndarray, dict]:
    """y_raw with each patient's forecast reviewed by the condition's critic;
    only patients with an out-of-range value are sent to it."""
    from llm_client import LocalLLM            # needs an Ollama server, so only here
    variant = split_condition(condition)[1]
    cv = critic_variant(run.cfg, variant)
    vcfg = cfg_for_llm_variant(run.cfg, variant)
    llm = LocalLLM(cfg_for_llm_variant(run.cfg, cv) if cv is not None else vcfg)
    critic = CriticAgent(llm, vcfg, enabled=True)
    lo, hi = run.plausible[:, 0], run.plausible[:, 1]
    flagged = np.where(np.any((y_raw < lo) | (y_raw > hi), axis=(1, 2)))[0]
    log.info("Critic (%s): %d of %d patients have out-of-range values; loading the model.",
             llm.model, len(flagged), len(y_raw))
    llm.load()

    def review(i):
        f = {"forecast": {v: y_raw[i, :, j].tolist() for j, v in enumerate(run.variables)}, "interval_halfwidth": None}
        r = critic.review(f)
        return i, np.stack([np.asarray(r["forecast"][v], dtype=float) for v in run.variables], axis=-1), r["clipped_values"]

    out, clipped = y_raw.copy(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, y, c in ex.map(review, flagged):
            out[i], clipped = y, clipped + c
    before = violations(run, y_raw)
    return out, {"model": llm.model, "patients": int(len(y_raw)), "patients_flagged": int(len(flagged)),
                 "values_out_of_range": int(before.sum()), "fixed_by_llm": int(before.sum() - clipped),
                 "clipped_after_llm": int(clipped), "failed_calls": dict(critic.error_counts)}


def critic_corrections(run: FinishedRun, condition: str, cal: dict, test: dict, out_dir: Path, workers: int) -> dict:
    """{"cal": reviewed y_raw, "test": ..., "report": ...}, from the cache if it
    matches these patients, else from the LLM (and cached)."""
    path = out_dir / f"llm_critic_{condition}.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["cal_ids"], cal["stay_ids"]) and np.array_equal(z["test_ids"], test["stay_ids"]):
            log.info("LLM critic corrections from %s (no LLM calls).", path)
            return {"cal": z["cal"], "test": z["test"], "report": json.loads(str(z["report"]))}
        log.warning("%s is for other patients; reviewing again.", path)
    cal_y, cal_rep = llm_review(run, condition, cal["y_raw"], workers)
    test_y, test_rep = llm_review(run, condition, test["y_raw"], workers)
    report = {"calibration": cal_rep, "test": test_rep}
    np.savez_compressed(path, cal=cal_y, test=test_y, cal_ids=cal["stay_ids"], test_ids=test["stay_ids"],
                        report=json.dumps(report))
    return {"cal": cal_y, "test": test_y, "report": report}


def score(run: FinishedRun, y_true: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          rng: np.random.RandomState) -> dict:
    e = per_patient_smape(y_true, pred)
    point, ci_lo, ci_hi = bootstrap_ci(e, run.cfg["evaluation"]["bootstrap_resamples"],
                                       run.cfg["evaluation"]["bootstrap_alpha"], rng)
    v = violations(run, pred)
    return {"smape": point, "smape_ci": [ci_lo, ci_hi], "coverage_calibrated": interval_coverage(y_true, lo, hi),
            "final_out_of_range": int(v.sum()), "final_out_of_range_rate": float(v.sum() / np.sum(~np.isnan(pred))),
            "per_patient_smape": e}


def paired(a: np.ndarray, b: np.ndarray, run: FinishedRun, rng: np.random.RandomState) -> dict:
    d = a - b
    d = d[~np.isnan(d)]
    point, lo, hi = bootstrap_ci(d, run.cfg["evaluation"]["bootstrap_resamples"],
                                 run.cfg["evaluation"]["bootstrap_alpha"], rng)
    return {"n": int(len(d)), "mean_diff": point, "ci": [lo, hi], "share_a_better": float(np.mean(d < 0)),
            "wilcoxon_pvalue": float(wilcoxon(d).pvalue) if np.any(d != 0) else None}


def replay_final(run: FinishedRun, condition: str, test: dict) -> float:
    """Largest |difference| between the run's test forecasts and y_raw
    re-post-processed with the run's settings and clipped (stage final)."""
    pp = run.calibration["conditions"][condition]["postprocess"]
    run.cfg["critic_agent"]["stage"] = "final"
    try:
        pred = run.postprocess(condition, test["y_raw"], test["obs"], pp["drift_damping"], pp["level_anchor_weight"])
    finally:
        run.cfg["critic_agent"]["stage"] = "raw"
    return float(np.nanmax(np.abs(pred - test["y_pred"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the finished run's tuned config")
    parser.add_argument("--condition", default="full_pipeline")
    parser.add_argument("--fit", default="global/standard,vitals_labs/extended",
                        help="comma-separated <strategy>/<grid> post-processing fits")
    parser.add_argument("--out-dir", default=None, help="default <results_dir>/critic_replay")
    parser.add_argument("--offline-only", action="store_true", help="skip the LLM critic arm (no GPU)")
    args = parser.parse_args()

    fits = [tuple(f.split("/")) for f in args.fit.split(",")]
    for s, g in fits:
        if s not in STRATEGIES or g not in GRIDS:
            parser.error(f"--fit {s}/{g}: strategy must be one of {STRATEGIES}, grid one of {tuple(GRIDS)}")
    run = FinishedRun(args.config)
    c = args.condition
    if split_condition(c)[0] != "full_pipeline":
        parser.error("--condition must be a full_pipeline condition (the one with a critic)")
    if (run.cfg.get("critic_agent") or {}).get("stage", "final") != "final":
        parser.error("this run's critic already reviewed the raw forecasts; its y_raw is post-critic")
    out_dir = Path(args.out_dir) if args.out_dir else run.results_dir / "critic_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = 1 - run.cfg["evaluation"]["interval_width_pct"] / 100
    rng = np.random.RandomState(run.cfg.get("random_seed", 42))

    cal, test = run.calibration_forecasts(c), run.test_forecasts(c)
    if cal["y_raw"] is None or "y_raw" not in test:
        raise SystemExit(f"{c}: no saved y_raw in this run")
    check = replay_final(run, c, test)          # also leaves critic_agent.stage = raw for the arms
    log.info("Replay of the run's own %s from y_raw: max |difference| %.2e", c, check)

    reviewed = {"no_critic": (cal["y_raw"], test["y_raw"]),
                "clip_raw": tuple(np.clip(y, run.plausible[:, 0], run.plausible[:, 1]) for y in (cal["y_raw"], test["y_raw"]))}
    critic_report = None
    if not args.offline_only:
        workers = run.cfg["performance"]["llm_max_concurrent_requests"]
        corr = critic_corrections(run, c, cal, test, out_dir, workers)
        reviewed["llm_critic_raw"], critic_report = (corr["cal"], corr["test"]), corr["report"]

    cal_half = (cal["y_upper"] - cal["y_lower"]) / 2
    test_half = (test["y_upper"] - test["y_lower"]) / 2
    report = {"config": args.config, "condition": c, "replay_check_max_abs_diff": check,
              "out_of_range_before_critic": {
                  "calibration": dict(zip(run.variables, violations(run, cal["y_raw"]).tolist())),
                  "test": dict(zip(run.variables, violations(run, test["y_raw"]).tolist()))},
              "llm_critic": critic_report, "fits": {}}

    for strategy, grid in fits:
        key = f"{strategy}/{grid}"
        arms, scores = {}, {}
        for arm, (cal_y, test_y) in reviewed.items():
            lam, w = fit(run, c, cal_y, cal["obs"], cal["y_true"], strategy, grid)
            cal_pred = run.postprocess(c, cal_y, cal["obs"], lam, w)
            factors = {v: conformal_factor(cal["y_true"][..., i], cal_pred[..., i], cal_half[..., i], alpha)
                       for i, v in enumerate(run.variables)}
            pred = run.postprocess(c, test_y, test["obs"], lam, w)
            lo, hi = apply_factors(pred, pred - test_half, pred + test_half, factors, run.variables)
            np.savez_compressed(out_dir / f"{arm}_{strategy}_{grid}_raw.npz", y_true=test["y_true"], y_pred=pred,
                                y_lower=pred - test_half, y_upper=pred + test_half, stay_ids=test["stay_ids"], y_raw=test_y)
            scores[arm] = score(run, test["y_true"], pred, lo, hi, rng)
            arms[arm] = {"settings": {v: [float(lam[i]), float(w[i])] for i, v in enumerate(run.variables)},
                         "factors": factors,
                         "calibration_smape": float(np.nanmean(per_patient_smape(cal["y_true"], cal_pred))),
                         **{k: v for k, v in scores[arm].items() if k != "per_patient_smape"}}
        # The run's own condition: critic after post-processing.
        src = run.results_dir / ("" if key == "global/standard" else f"rescored_{strategy}_{grid}") / f"{c}_raw.npz"
        if src.exists():
            with np.load(src) as z:
                f = {k: z[k] for k in z.files}
            if np.array_equal(f["stay_ids"], test["stay_ids"]):
                fac = run.factors(c) if key == "global/standard" else \
                    json.loads((src.parent / "rescore.json").read_text())["conditions"][c]["factors"]
                lo, hi = apply_factors(f["y_pred"], f["y_lower"], f["y_upper"], fac, run.variables)
                scores["critic_final"] = score(run, f["y_true"], f["y_pred"], lo, hi, rng)
                arms["critic_final"] = {"source": str(src),
                                        **{k: v for k, v in scores["critic_final"].items() if k != "per_patient_smape"}}
        comps = {f"{a} - {b}": paired(scores[a]["per_patient_smape"], scores[b]["per_patient_smape"], run, rng)
                 for a, b in COMPARISONS if a in scores and b in scores}
        report["fits"][key] = {"arms": arms, "comparisons": comps}

        print(f"\n== {c}, post-processing fit {key} (test, {len(test['stay_ids'])} patients) ==")
        print(f"{'arm':16s} {'sMAPE':>8s} {'95% CI':>17s} {'coverage':>8s} {'out of range':>12s}")
        for arm, a in arms.items():
            print(f"{arm:16s} {a['smape']:8.4f} [{a['smape_ci'][0]:.4f}, {a['smape_ci'][1]:.4f}] "
                  f"{a['coverage_calibrated']:8.3f} {a['final_out_of_range']:12d}")
        for name, r in comps.items():
            p = "n/a" if r["wilcoxon_pvalue"] is None else f"{r['wilcoxon_pvalue']:.2g}"
            print(f"  {name:32s} {r['mean_diff']:+.6f} [{r['ci'][0]:+.6f}, {r['ci'][1]:+.6f}]  p = {p}")

    if critic_report:
        for stage, r in critic_report.items():
            print(f"LLM critic, {stage}: {r['values_out_of_range']} values out of range in {r['patients_flagged']} of "
                  f"{r['patients']} patients; fixed by the LLM {r['fixed_by_llm']}, clipped {r['clipped_after_llm']}"
                  f"{'; failed calls ' + str(r['failed_calls']) if r['failed_calls'] else ''}")
    print(f"Replay check (run's own {c} from y_raw): max |difference| {check:.2e}")
    path = out_dir / f"critic_replay_{c}.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print("Written to", out_dir)


if __name__ == "__main__":
    main()
