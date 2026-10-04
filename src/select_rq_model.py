"""
Choose which model answers RQ1-RQ3, using validation data only.

Selection rule (fixed here, before any test result is seen):
  1. Candidates: llm.model (the pre-specified primary) and every LLM variant
     that has a `full_pipeline@<variant>` condition, except agent-mix
     variants (those setting critic_variant).
  2. Score: mean per-patient sMAPE of `full_pipeline` on the CALIBRATION
     subset of the validation split (written by src/calibrate.py); the test
     split is never read. Lower is better.
  3. Eligible: LLM fallback rate on those patients at or below the tuning
     grid's max_fallback_rate (a model can't win by falling back to the
     naive forecast).
  4. The best eligible candidate replaces the primary only if it is
     significantly better: the 95% paired-bootstrap CI of its per-patient
     sMAPE difference from the primary lies entirely below 0. Otherwise the
     primary stays.
The primary's RQ analysis remains the pre-specified one; a chosen model's
is reported alongside it as a secondary analysis (evaluation.rq_models).

Usage:
    python src/select_rq_model.py --config-file config/config_nibi_lean_tuned.yaml [--write-config]
--write-config adds the chosen model to evaluation.rq_models in that config
and the conditions its RQ analysis needs (full_pipeline_no_critic@<v>,
full_pipeline_no_similarity@<v>). Then compute them, e.g.
    bash jobs/run_all.sh lean --unattended --redo calibrate,run,evaluate
(checkpoints mean only the new conditions are computed).
Writes:
    <results_dir>/rq_model_selection.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from common import get_logger, load_config, parse_step_args, split_condition
from tune import DEFAULT_GRID, load_grid

log = get_logger("select_rq_model")

BOOTSTRAP = 2000


def candidates(cfg: dict, available: list[str]) -> dict[str, str]:
    """{model label: full_pipeline condition} for the rule's candidates."""
    variants = (cfg["llm"].get("variants") or {})
    out = {}
    for condition in available:
        base, variant = split_condition(condition)
        if base != "full_pipeline":
            continue
        if variant is None:
            out["primary"] = condition
        elif not variants.get(variant, {}).get("critic_variant"):
            out[variant] = condition
    return out


def main(config_path: str, write_config: bool) -> None:
    cfg = load_config(config_path)
    results_dir = Path(cfg["paths"]["results_dir"])
    smape_path, cal_path = results_dir / "calibration_smape.npz", results_dir / "calibration.json"
    if not smape_path.exists() or not cal_path.exists():
        raise SystemExit(f"{smape_path.name} / {cal_path.name} not found in {results_dir} — run "
                         "src/calibrate.py with this config first.")
    cal = json.loads(cal_path.read_text())["conditions"]
    with np.load(smape_path) as z:
        per_patient = {k: z[k] for k in z.files if k != "stay_ids"}
    cands = candidates(cfg, list(per_patient))
    if "primary" not in cands:
        raise SystemExit("No full_pipeline result for the primary model in the calibration results.")
    # The grid the config was tuned with (tuned_from.grid), else the default;
    # relative paths are from the project directory.
    grid = Path((cfg.get("tuned_from") or {}).get("grid") or DEFAULT_GRID)
    if not grid.is_absolute() and not grid.exists():
        grid = Path(__file__).resolve().parents[1] / grid
    max_fb = load_grid(str(grid))["max_fallback_rate"]
    rng = np.random.RandomState(cfg["cohort"]["random_seed"])
    primary = per_patient[cands["primary"]]

    rows = []
    for label, condition in cands.items():
        pp = per_patient[condition]
        fb = cal.get(condition, {}).get("llm_fallback_rate") or 0.0
        diff = pp - primary
        mask = ~np.isnan(diff)
        d = diff[mask]
        boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)] if len(d) else [np.nan]
        rows.append({
            "model": label, "condition": condition, "smape_mean": float(np.nanmean(pp)),
            "fallback_rate": fb, "eligible": fb <= max_fb,
            "diff_vs_primary": float(d.mean()) if len(d) else None,
            "diff_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_paired_patients": int(mask.sum()),
        })
    rows.sort(key=lambda r: r["smape_mean"])

    eligible = [r for r in rows if r["eligible"]]
    best = eligible[0] if eligible else None
    if best is None:
        chosen, reason = "primary", "no candidate is eligible (fallback rates too high); the primary stays"
    elif best["model"] == "primary":
        chosen, reason = "primary", "the primary has the lowest sMAPE among eligible models"
    elif best["diff_ci95"][1] < 0:
        chosen, reason = best["model"], (f"{best['model']} is significantly better than the primary "
                                         f"(difference {best['diff_vs_primary']:+.4f}, 95% CI "
                                         f"[{best['diff_ci95'][0]:+.4f}, {best['diff_ci95'][1]:+.4f}])")
    else:
        chosen, reason = "primary", (f"{best['model']} has the lowest sMAPE, but its difference from the "
                                     f"primary is not significant (95% CI [{best['diff_ci95'][0]:+.4f}, "
                                     f"{best['diff_ci95'][1]:+.4f}]); the primary stays")

    print(f"\nfull_pipeline on {rows[0]['n_paired_patients']} calibration patients "
          f"(max fallback rate {max_fb:.0%}):")
    print(f"{'model':18s} {'sMAPE':>7s} {'vs primary':>11s} {'95% CI':>21s} {'fallbacks':>9s}")
    for r in rows:
        ci = f"[{r['diff_ci95'][0]:+.4f}, {r['diff_ci95'][1]:+.4f}]" if r["model"] != "primary" else ""
        print(f"{r['model']:18s} {r['smape_mean']:7.4f} {r['diff_vs_primary']:+11.4f} {ci:>21s} "
              f"{r['fallback_rate']:8.1%}{'' if r['eligible'] else ' (ineligible)'}")
    print(f"\nChosen for RQ1-RQ3: {chosen} — {reason}.")

    out = {"rule": __doc__.split("Usage:")[0].strip(), "chosen": chosen, "reason": reason,
           "max_fallback_rate": max_fb, "candidates": rows, "config": config_path}
    (results_dir / "rq_model_selection.json").write_text(json.dumps(out, indent=2))
    log.info("Selection written to %s", results_dir / "rq_model_selection.json")

    if write_config and chosen != "primary":
        add_to_config(config_path, chosen)


def add_to_config(config_path: str, variant: str) -> None:
    """Add the variant to evaluation.rq_models and the conditions its RQ
    analysis needs. Keeps the file's leading comment lines (the tuned
    config's header); the rest is rewritten by yaml."""
    path = Path(config_path)
    text = path.read_text()
    header = ""
    for ln in text.splitlines(keepends=True):
        if not ln.startswith("#"):
            break
        header += ln
    raw = yaml.safe_load(text)
    conds = raw["conditions"]
    added = []
    for base in ("single_model_llm", "full_pipeline", "full_pipeline_no_critic", "full_pipeline_no_similarity"):
        c = f"{base}@{variant}"
        if c not in conds:
            conds.append(c)
            added.append(c)
    ev = raw.setdefault("evaluation", {})
    models = ev.get("rq_models") or ["primary"]
    if variant not in models:
        models.append(variant)
    ev["rq_models"] = models
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    log.info("%s: rq_models = %s; conditions added: %s", path, models, added or "none")
    print(f"\nUpdated {path}: rq_models = {models}; conditions added: {added or 'none'}.\n"
          "Compute them with:  bash jobs/run_all.sh lean --unattended --redo calibrate,run,evaluate")


if __name__ == "__main__":
    def _add(p):
        p.add_argument("--write-config", action="store_true",
                       help="add the chosen model and its RQ conditions to this config")
    args = parse_step_args(__doc__.strip().splitlines()[0], add_arguments=_add)
    main(args.config_file, args.write_config)
