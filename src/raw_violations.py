"""
Plausibility violations in the LLM's own forecasts, before post-processing,
counted from a finished run's checkpoints — no LLM calls, no GPU; light
enough for a login node.

The critic reviews the post-processed forecast unless critic_agent.stage is
raw, and the level anchor and damping keep that close to the observed values
(tri_lean_exp3.4: 0 violations in every condition). The checkpoints keep each
LLM forecast before post-processing (y_raw), so this shows what the critic
would have had to fix had it reviewed the model's own output. Newer runs
save y_raw in <results_dir>/*_raw.npz too, and evaluate_results.py reports
it (<condition>_before_postprocess); this is for runs from before that.

Usage:
    python src/raw_violations.py <checkpoint dir> [<checkpoint dir> ...]
e.g. $DT_RESULTS_DIR/mimic-iv-twin-work/tri_lean_exp3.4/checkpoints_tuned
(both run_experiment/ and calibration/ under it are read). The plausible
ranges are those in each condition's fingerprint.json, i.e. the config the
forecasts were made with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def condition_violations(cdir: Path) -> dict | None:
    """Counts over one condition checkpoint's batches, or None if they hold
    no y_raw (not an LLM condition)."""
    fp = json.loads((cdir / "fingerprint.json").read_text())["fingerprint"]
    names = [v["name"] for v in fp["variables"]]
    lo = np.array([v["plausible_range"][0] for v in fp["variables"]], dtype=float)
    hi = np.array([v["plausible_range"][1] for v in fp["variables"]], dtype=float)
    cells = np.zeros(len(names), dtype=int)
    below = np.zeros(len(names), dtype=int)
    above = np.zeros(len(names), dtype=int)
    patients = patients_any = 0
    found = False
    for path in sorted(cdir.glob("batch_*.npz")):
        with np.load(path) as z:
            if "y_raw" not in z.files:
                continue
            y = z["y_raw"]                       # (patients, horizon, variables)
        found = True
        finite = np.isfinite(y)
        lo_out, hi_out = finite & (y < lo), finite & (y > hi)
        cells += finite.sum(axis=(0, 1))
        below += lo_out.sum(axis=(0, 1))
        above += hi_out.sum(axis=(0, 1))
        patients += y.shape[0]
        patients_any += int((lo_out | hi_out).any(axis=(1, 2)).sum())
    if not found:
        return None
    return {"patients": patients, "patients_with_violation": patients_any,
            "cells": int(cells.sum()), "violations": int((below + above).sum()),
            "violation_rate": float((below + above).sum() / max(1, cells.sum())),
            "per_variable": {n: {"below": int(b), "above": int(a), "rate": float((a + b) / max(1, c))}
                             for n, b, a, c in zip(names, below, above, cells)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write the counts to this JSON file")
    args = parser.parse_args()

    out = {}
    for root in args.checkpoint_dirs:
        for cdir in sorted(root.glob("**/conditions/*")):
            if not (cdir / "fingerprint.json").exists():
                continue
            stage = cdir.parent.parent.name     # run_experiment, calibration, tune
            record = condition_violations(cdir)
            if record is not None:
                out[f"{stage}/{cdir.name}"] = record

    if not out:
        print("No LLM condition checkpoints with y_raw found under", *args.checkpoint_dirs)
        return
    names = list(next(iter(out.values()))["per_variable"])
    print(f"{'stage/condition':58s} {'patients':>8s} {'pts w/ viol':>11s} {'rate':>8s}  "
          + "  ".join(f"{n:>14s}" for n in names))
    for key, r in out.items():
        per = "  ".join(f"{pv['below']:>6d}<{pv['above']:>6d}>" for pv in r["per_variable"].values())
        print(f"{key:58s} {r['patients']:8d} {r['patients_with_violation']:11d} {r['violation_rate']:8.4%}  {per}")
    print("\nPer variable: values below '<' and above '>' the plausible range, before post-processing.")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print("Written to", args.json)


if __name__ == "__main__":
    main()
