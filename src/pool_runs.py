"""
Paired comparisons pooled over several runs (independent samples of
patients): per sample, pooled, and within the RQ3 subgroups (stable vs
deteriorating). Reads only the saved forecasts (<results_dir>/*_raw.npz) and
each run's statistical_analysis.json, so no LLM calls and no GPU; light
enough for a login node.

One run answers each comparison on 450 test patients; tri_lean_exp3.4,
tri_lean_exp4 and ror_lean_exp5 draw three different samples (cohort seeds
42, 43, 44), so their per-patient differences can be pooled. For each
comparison [A, B] (A - B per-patient sMAPE, as in evaluate_results.py):

  - per sample: mean difference and Wilcoxon signed-rank p;
  - pooled: the same over all samples' patients, with a patient-bootstrap
    95 % CI of the mean difference;
  - subgroups (RQ3): the pooled test among stable and among deteriorating
    patients (vasopressor started in the forecast horizon, the lists in each
    run's statistical_analysis.json), and a Mann-Whitney U test of whether
    the differences differ between the two (subgroup x comparison
    interaction).

A sample may take a condition from another run (`overrides`), e.g.
tri_lean_exp3.4's LSTM from tri_lean_exp3.4_lstm, which re-trained it with
the current lstm_input on the same patients; the two runs' stay_ids must
match. An ensemble "X+Y" is the mean of its members' forecasts, as
pipeline.ensemble_result builds it: its saved forecasts are used when no
member is overridden, and it is rebuilt from the members otherwise.

Usage:
    python src/pool_runs.py config/pooled_lean.yaml --root <dir> [--json <file>]
<dir> holds each run's results as jobs/pack_run.sh packs them (untar each
<run>_results.tar into <dir>/<run>/); the spec's paths are relative to it.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu, wilcoxon

from evaluate_results import bootstrap_ci, per_patient_smape

SUBGROUPS = ("stable", "deteriorating")


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in ("stay_ids", "y_true", "y_pred")}


def condition_forecasts(sample: dict, root: Path, condition: str) -> dict:
    """stay_ids, y_true and y_pred of one condition in one sample."""
    results = root / sample["results_dir"]
    overrides = {c: root / d for c, d in (sample.get("overrides") or {}).items()}
    members = condition.split("+")
    saved = results / f"{condition}_raw.npz"
    if len(members) == 1 or (saved.exists() and not any(m in overrides for m in members)):
        return _load(overrides.get(condition, results) / f"{condition}_raw.npz")
    parts = [condition_forecasts(sample, root, m) for m in members]
    for p in parts[1:]:
        if not np.array_equal(p["stay_ids"], parts[0]["stay_ids"]):
            raise ValueError(f"{sample['name']}: members of '{condition}' forecast different patients")
    return {"stay_ids": parts[0]["stay_ids"], "y_true": parts[0]["y_true"],
            "y_pred": np.mean([p["y_pred"] for p in parts], axis=0)}


def deteriorating_ids(sample: dict, root: Path) -> set[int]:
    path = root / sample.get("deteriorating_from", Path(sample["results_dir"]) / "statistical_analysis.json")
    rq3 = json.loads(path.read_text())["RQ3_deterioration_subgroup"]
    return {int(s) for s in rq3["deteriorating_stay_ids"]}


def _wilcoxon_p(d: np.ndarray) -> float | None:
    return float(wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else None


def _summary(d: np.ndarray) -> dict:
    return {"n": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else None,
            "share_a_better": float(np.mean(d < 0)) if len(d) else None, "wilcoxon_pvalue": _wilcoxon_p(d)}


def compare(spec: dict, root: Path, a: str, b: str, rng: np.random.RandomState) -> dict:
    per_sample, pooled = {}, {"all": [], **{g: [] for g in SUBGROUPS}}
    for sample in spec["samples"]:
        fa, fb = condition_forecasts(sample, root, a), condition_forecasts(sample, root, b)
        if not np.array_equal(fa["stay_ids"], fb["stay_ids"]):
            raise ValueError(f"{sample['name']}: '{a}' and '{b}' forecast different patients")
        d = per_patient_smape(fa["y_true"], fa["y_pred"]) - per_patient_smape(fb["y_true"], fb["y_pred"])
        det = np.array([int(s) in deteriorating_ids(sample, root) for s in fa["stay_ids"]])
        keep = ~np.isnan(d)
        groups = {"all": d[keep], "stable": d[keep & ~det], "deteriorating": d[keep & det]}
        per_sample[sample["name"]] = {g: _summary(v) for g, v in groups.items()}
        for g, v in groups.items():
            pooled[g].append(v)
    pooled = {g: np.concatenate(v) for g, v in pooled.items()}
    _, lo, hi = bootstrap_ci(pooled["all"], spec.get("bootstrap_resamples", 2000),
                             spec.get("bootstrap_alpha", 0.05), rng)
    s, t = pooled["stable"], pooled["deteriorating"]
    return {
        "comparison": f"{a}_vs_{b}",
        "per_sample": per_sample,
        "pooled": {**_summary(pooled["all"]), "mean_diff_ci95": [lo, hi]},
        "subgroups": {g: _summary(pooled[g]) for g in SUBGROUPS},
        "subgroup_interaction_mannwhitney_pvalue":
            float(mannwhitneyu(s, t).pvalue) if len(s) and len(t) else None,
    }


def shared_test_patients(spec: dict, root: Path, condition: str) -> dict:
    """Test patients two samples have in common (independent draws from one
    eligible pool overlap a little by chance)."""
    ids = {s["name"]: set(condition_forecasts(s, root, condition)["stay_ids"].tolist()) for s in spec["samples"]}
    return {f"{x}&{y}": len(ids[x] & ids[y]) for x, y in combinations(ids, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path, help="YAML: samples and comparisons (e.g. config/pooled_lean.yaml)")
    parser.add_argument("--root", type=Path, required=True, help="directory the spec's paths are relative to")
    parser.add_argument("--json", type=Path, help="also write the results to this JSON file")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    rng = np.random.RandomState(spec.get("seed", 42))
    out = {"spec": str(args.spec), "samples": [s["name"] for s in spec["samples"]],
           "shared_test_patients": shared_test_patients(spec, args.root, spec["comparisons"][0][1]),
           "comparisons": [compare(spec, args.root, a, b, rng) for a, b in spec["comparisons"]]}

    fmt = lambda r: (f"{r['mean_diff']:+.4f} (p={r['wilcoxon_pvalue']:.2g})"
                     if r["n"] and r["wilcoxon_pvalue"] is not None else "n/a")
    print("Per-patient sMAPE difference A - B (negative: A better); Wilcoxon signed-rank p.")
    print("Test patients shared between samples:", out["shared_test_patients"])
    for c in out["comparisons"]:
        p, sg = c["pooled"], c["subgroups"]
        print(f"\n{c['comparison']}")
        print("   " + "; ".join(f"{n}: {fmt(r['all'])}" for n, r in c["per_sample"].items()))
        print(f"   pooled n={p['n']}: {fmt(p)}, 95% CI [{p['mean_diff_ci95'][0]:+.4f}, {p['mean_diff_ci95'][1]:+.4f}], "
              f"A better for {p['share_a_better']:.0%} of patients")
        print(f"   stable n={sg['stable']['n']}: {fmt(sg['stable'])} | deteriorating n={sg['deteriorating']['n']}: "
              f"{fmt(sg['deteriorating'])} | interaction p={c['subgroup_interaction_mannwhitney_pvalue']:.2g}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print("\nWritten to", args.json)


if __name__ == "__main__":
    main()
