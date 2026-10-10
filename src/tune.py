"""
Hyperparameter search on the validation split — sequential (coordinate)
search over the settings in a grid file, run before the final experiment.

The validation partition is split once, with a fixed seed, into
  - a TUNING subset (tuning.n_tune_patients, default 128), on which every
    trial is scored here, and
  - a CALIBRATION subset (the rest), kept untouched for src/calibrate.py.
The test partition is never read.

The grid file lists rounds. Each round evaluates the current best settings
(the "incumbent") and every trial of the round (the incumbent plus that
trial's overrides) on the round's conditions, and keeps a trial only if it
lowers the objective — mean per-patient sMAPE over the round's conditions —
by at least tuning.min_improvement, with every condition's LLM fallback rate
at or below tuning.max_fallback_rate (so a setting can't win by producing
naive fallbacks). The winner becomes the incumbent for the next round.
A round with `per_condition: true` (prompt settings) instead picks a winner
for each of its conditions, scored alone, and sets it for that condition's
prompt group only (forecasting_agent.per_condition, see common.py), so the
single model and the pipeline can each keep the prompt that suits them.

Resuming: every condition × setting is checkpointed per batch of patients
under <checkpoint_dir>/tune/ (see src/checkpoint.py), and its metrics are
cached once finished, so a stopped run continues where it left off and a
setting evaluated in an earlier round is never recomputed. --full-refresh
deletes the tuning checkpoints and starts over.

Usage:
    python src/tune.py --config-file config/config_alliance_lean.yaml [--grid config/tuning_grid.yaml] [--full-refresh]
Writes:
    <results_dir>/tuning/trials.csv        every trial's metrics (rewritten after each trial)
    <results_dir>/tuning/rounds.json       each round's incumbent, candidates and winner
    <results_dir>/tuning/tuned_overrides.yaml
    config/<config stem>_tuned.yaml        the base config with the winning overrides, writing
                                           to its own results/checkpoint directories
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import tracking
from checkpoint import RunCheckpoint, atomic_path, checkpoint_root, condition_fingerprint, model_fingerprint
from common import (LLM_CONDITIONS, PER_CONDITION_KEYS, ensure_work_dirs, get_logger, load_config, parse_step_args,
                    prompt_group, split_condition)
from evaluate_results import per_patient_smape
from harmonization_agent import build_tensors
from metrics import interval_coverage, mean_interval_width, plausibility_violation_rate
from pipeline import fit_models, run_condition, validate_conditions
from run_experiment import load_cohort_and_panel, split_tensors

log = get_logger("tune")

DEFAULT_GRID = "config/tuning_grid.yaml"


# ---------------------------------------------------------------------------
# Shared with src/calibrate.py
# ---------------------------------------------------------------------------

def config_grid(cfg: dict) -> str:
    """The config's tuning_grid (e.g. a grid without rounds, to run with
    settings fixed in the config), else DEFAULT_GRID."""
    return cfg.get("tuning_grid") or DEFAULT_GRID


def load_grid(path: str) -> dict:
    grid = yaml.safe_load(Path(path).read_text())["tuning"]
    grid.setdefault("n_tune_patients", 128)
    grid.setdefault("seed", 42)
    grid.setdefault("max_fallback_rate", 0.10)
    grid.setdefault("min_improvement", 0.0)
    grid.setdefault("fixed", {})
    grid.setdefault("rounds", [])
    return grid


def validation_subsets(val_ids, n_tune: int, seed: int) -> tuple[list[int], list[int]]:
    """Seeded, disjoint split of the validation stays into (tuning, calibration)."""
    ids = sorted(int(i) for i in val_ids)
    order = np.random.RandomState(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n_tune = min(n_tune, len(ids))
    return sorted(shuffled[:n_tune]), sorted(shuffled[n_tune:])


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """A copy of cfg with each dotted key (e.g. 'similarity_agent.k_neighbors') set."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return out


def config_for_conditions(cfg: dict, conditions: list[str]) -> dict:
    """cfg restricted to `conditions`, building only the models they need."""
    out = copy.deepcopy(cfg)
    out["conditions"] = list(conditions)
    # Ensembles are built from finished conditions and never tuned: keep
    # only those whose members are all among these conditions.
    out["ensembles"] = [m for m in cfg.get("ensembles") or [] if all(c in conditions for c in m)]
    bases = {split_condition(c)[0] for c in conditions}
    out["baselines"]["run_gbm"] = "gbm" in bases
    out["baselines"]["run_lstm"] = "lstm" in bases
    out["baselines"]["run_single_model_llm"] = "single_model_llm" in bases
    validate_conditions(out)
    return out


def condition_metrics(result: dict, cfg: dict) -> dict:
    """Scalar metrics of one condition's forecasts on the patients it ran on."""
    variables = [v["name"] for v in cfg["variables"]]
    pp = per_patient_smape(result["y_true"], result["y_pred"])
    n = len(result["stay_ids"])
    pvr = plausibility_violation_rate(result["y_pred"], variables)
    fallbacks = result["llm_fallbacks"]
    filled = result.get("llm_filled")
    return {
        "n_patients": n,
        "smape_mean": float(np.nanmean(pp)),
        "smape_median": float(np.nanmedian(pp)),
        "fallback_rate": (fallbacks / max(1, n)) if fallbacks is not None else 0.0,
        # Share of (patient, variable) forecasts filled with the naive forecast
        # because the LLM left the variable out. Reported, not a constraint:
        # the naive values are scored like any other forecast.
        "filled_rate": float(np.sum(filled)) / max(1, n * len(variables)) if filled is not None else 0.0,
        "interval_coverage": interval_coverage(result["y_true"], result["y_lower"], result["y_upper"]),
        # Per variable (each in its own unit), one trials.csv column each.
        **{f"interval_width_{var}": mean_interval_width(result["y_lower"][..., i], result["y_upper"][..., i])
           for i, var in enumerate(variables)},
        "plausibility_violation_rate": float(np.nanmean(list(pvr.values()))),
    }


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TrialRunner:
    """Evaluates (settings, conditions) on the tuning subset, caching each
    condition's metrics by its checkpoint fingerprint."""

    def __init__(self, base_cfg: dict, splits: dict, tune_ids: list[int], root: Path):
        self.base_cfg = base_cfg
        self.train = splits["train"]
        self.train_ids = sorted(self.train)
        self.tune_ids = tune_ids
        self.eval_tensors = {sid: splits["val"][sid] for sid in tune_ids}
        self.root = root

    def run(self, overrides: dict, conditions: list[str]) -> dict[str, dict]:
        cfg = config_for_conditions(apply_overrides(self.base_cfg, overrides), conditions)
        fitted = None
        out = {}
        for condition in conditions:
            fp = condition_fingerprint(cfg, condition, self.train_ids, self.tune_ids)
            cdir = self.root / "conditions" / f"{condition}__{_digest(fp)}"
            metrics_path = cdir / "metrics.json"
            if metrics_path.exists():
                out[condition] = json.loads(metrics_path.read_text())
                continue
            if fitted is None:
                models_fp = {m: model_fingerprint(cfg, m, self.train_ids) for m in ("gbm", "lstm")}
                model_ckpt = RunCheckpoint(self.root / "models" / _digest(models_fp))
                model_ckpt.claim()
                fitted = fit_models(cfg, self.train, checkpoint=model_ckpt)
            ckpt = RunCheckpoint(cdir)
            ckpt.claim()
            result = run_condition(condition, cfg, self.train, self.eval_tensors, fitted,
                                   checkpoint=ckpt.condition(condition, fp),
                                   progress=tracking.progress_logger("tune/"))
            tracking.log_errors(f"tune/{condition}", result["llm_errors"])
            metrics = condition_metrics(result, cfg) | {"condition": condition}
            ckpt.check_owner()
            with atomic_path(metrics_path) as tmp:
                tmp.write_text(json.dumps(metrics, indent=2))
            out[condition] = metrics
        return out


def objective(metrics: dict[str, dict], max_fallback_rate: float) -> tuple[float, bool]:
    """(mean per-patient sMAPE over the conditions, eligible)."""
    value = float(np.mean([m["smape_mean"] for m in metrics.values()]))
    eligible = all(m["fallback_rate"] <= max_fallback_rate for m in metrics.values())
    return value, eligible


def group_overrides(overrides: dict, group: str) -> dict:
    """A per_condition round's trial overrides (forecasting_agent prompt
    settings) for one prompt group only: forecasting_agent.per_condition.<group>.<key>."""
    out = {}
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if section != "forecasting_agent" or key not in PER_CONDITION_KEYS:
            raise ValueError(f"per_condition round: '{dotted}' is not a forecasting_agent prompt setting "
                             f"({', '.join(PER_CONDITION_KEYS)})")
        out[f"forecasting_agent.per_condition.{group}.{key}"] = value
    return out


def pick_winner(scored: dict, min_improvement: float) -> str:
    """The acceptance rule: the best eligible trial replaces the incumbent
    if it gains at least min_improvement (or the incumbent is ineligible)."""
    inc_value, inc_eligible = scored["incumbent"][:2]
    eligible = {t: s for t, s in scored.items() if s[1]}
    if eligible:
        best = min(eligible, key=lambda t: eligible[t][0])
        if best != "incumbent" and (not inc_eligible or inc_value - eligible[best][0] >= min_improvement):
            return best
    return "incumbent"


def tuned_config_path(config_path: str) -> Path:
    p = Path(config_path)
    return p.with_name(f"{p.stem}_tuned{p.suffix}")


def write_tuned_config(config_path: str, grid_path: str, overrides: dict, grid: dict) -> Path:
    """The base config file (unexpanded, so $DT_RESULTS_DIR etc. still resolve per
    environment) with the winning overrides, writing to its own results and
    checkpoint directories so the untuned run's outputs are kept."""
    raw = yaml.safe_load(Path(config_path).read_text())
    tuned = apply_overrides(raw, overrides)
    paths = tuned["paths"]
    paths["results_dir"] = str(paths["results_dir"]).rstrip("/") + "_tuned"
    paths["checkpoint_dir"] = str(paths["work_dir"]).rstrip("/") + "/checkpoints_tuned"
    tuned["tuned_from"] = {"config": config_path, "grid": grid_path, "overrides": overrides,
                           "n_tune_patients": grid["n_tune_patients"], "seed": grid["seed"]}
    out = tuned_config_path(config_path)
    header = (f"# GENERATED by src/tune.py from {config_path} and {grid_path} — do not edit by hand;\n"
              "# re-run tuning instead. Identical to the base config except for the overrides\n"
              "# listed under tuned_from, and results_dir / checkpoint_dir, which point to\n"
              "# separate *_tuned locations so the untuned run's outputs are kept.\n")
    # On clusters whose compute nodes can't write the repo (Trillium), out is a
    # symlink into $DT_RESULTS_DIR/tuned-configs, or under the tuned_configs link
    # to it for a run_all.sh run (jobs/setup_bash.sh); write through it.
    out.resolve().parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(tuned, sort_keys=False, allow_unicode=True))
    return out


def main(config_path: str, grid_path: str | None = None, full_refresh: bool = False) -> None:
    cfg = load_config(config_path)
    grid_path = grid_path or config_grid(cfg)
    grid = load_grid(grid_path)
    ensure_work_dirs(cfg)

    root = checkpoint_root(cfg) / "tune"
    if full_refresh:
        log.info("--full-refresh: deleting tuning checkpoints in %s.", root)
        RunCheckpoint(root).clear()
    out_dir = Path(cfg["paths"]["results_dir"]) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    tune_ids, calib_ids = validation_subsets(splits["val"], grid["n_tune_patients"], grid["seed"])
    log.info("Validation split: %d tuning patients (used here), %d calibration patients (kept for "
             "calibrate.py). The test split is not used.", len(tune_ids), len(calib_ids))

    tracking.start(cfg, "tune", config_path, {"n_tune_patients": len(tune_ids), "n_rounds": len(grid["rounds"])})
    runner = TrialRunner(cfg, splits, tune_ids, root)
    incumbent = dict(grid["fixed"])
    n_trials = 0
    trial_rows, round_log = [], []

    for rnd in grid["rounds"]:
        name, conditions = rnd["name"], rnd["conditions"]
        candidates = {"incumbent": {}} | (rnd.get("trials") or {})
        if rnd.get("per_condition"):
            # Each condition's prompt group gets its own winner: the trials
            # set forecasting_agent.per_condition.<group>.* and each
            # condition is scored alone. Trials already run with the same
            # effective settings are reused from the cache.
            groups = {c: prompt_group(split_condition(c)[0]) for c in conditions}
            if any(split_condition(c)[0] not in LLM_CONDITIONS for c in conditions) \
                    or len(set(groups.values())) != len(conditions):
                raise ValueError(f"Round '{name}': per_condition needs LLM conditions from different "
                                 f"prompt groups, got {conditions}")
            log.info("=== Round '%s' (per condition) on %s: %d settings ===", name, conditions, len(candidates))
            winners, entry_candidates, entry_objective, entry_before = {}, {}, {}, {}
            for condition in conditions:
                scored = {}
                for trial, trial_over in candidates.items():
                    overrides = incumbent | group_overrides(trial_over or {}, groups[condition])
                    metrics = runner.run(overrides, [condition])
                    value, eligible = objective(metrics, grid["max_fallback_rate"])
                    scored[trial] = (value, eligible, trial_over or {})
                    log.info("Round '%s' %s trial '%s': objective %.4f%s %s", name, condition, trial, value,
                             "" if eligible else " (INELIGIBLE: fallback rate too high)", trial_over or "")
                    n_trials += 1
                    tracking.log_metrics({"tune/trials_done": n_trials,
                                          f"tune/{name}/{condition}/{trial}/objective": value})
                    trial_rows.append({"round": name, "trial": trial, "objective": value, "eligible": eligible,
                                       "overrides": json.dumps(trial_over or {}), **metrics[condition]})
                    pd.DataFrame(trial_rows).to_csv(out_dir / "trials.csv", index=False)
                winners[condition] = pick_winner(scored, grid["min_improvement"])
                entry_candidates[condition] = {t: {"objective": s[0], "eligible": s[1]} for t, s in scored.items()}
                entry_objective[condition] = scored[winners[condition]][0]
                entry_before[condition] = scored["incumbent"][0]
                log.info("Round '%s' winner for %s: %s (objective %.4f; before %.4f)", name, condition,
                         winners[condition], entry_objective[condition], entry_before[condition])
                tracking.log_metrics({f"tune/{name}/{condition}/winner_objective": entry_objective[condition]})
            for condition, trial in winners.items():
                incumbent = incumbent | group_overrides(candidates[trial] or {}, groups[condition])
            round_log.append({"round": name, "conditions": conditions, "per_condition": True,
                              "winner": winners, "objective": entry_objective, "incumbent_before": entry_before,
                              "candidates": entry_candidates, "settings_after": incumbent})
            (out_dir / "rounds.json").write_text(json.dumps(round_log, indent=2))
            tracking.log_metrics({"tune/rounds_done": len(round_log)})
            continue
        log.info("=== Round '%s' on %s: %d settings ===", name, conditions, len(candidates))
        scored = {}
        for trial, trial_over in candidates.items():
            overrides = incumbent | (trial_over or {})
            metrics = runner.run(overrides, conditions)
            value, eligible = objective(metrics, grid["max_fallback_rate"])
            scored[trial] = (value, eligible, overrides)
            log.info("Round '%s' trial '%s': objective %.4f%s %s", name, trial, value,
                     "" if eligible else " (INELIGIBLE: fallback rate too high)", trial_over or "")
            n_trials += 1
            tracking.log_metrics({"tune/trials_done": n_trials, f"tune/{name}/{trial}/objective": value,
                                  f"tune/{name}/{trial}/eligible": int(eligible),
                                  **{f"tune/{name}/{trial}/{c}/fallback_rate": m["fallback_rate"]
                                     for c, m in metrics.items()}})
            for condition, m in metrics.items():
                trial_rows.append({"round": name, "trial": trial, "objective": value, "eligible": eligible,
                                   "overrides": json.dumps(trial_over or {}), **m})
            pd.DataFrame(trial_rows).to_csv(out_dir / "trials.csv", index=False)

        inc_value = scored["incumbent"][0]
        winner = pick_winner(scored, grid["min_improvement"])
        incumbent = scored[winner][2]
        round_log.append({"round": name, "conditions": conditions, "winner": winner,
                          "objective": scored[winner][0], "incumbent_before": inc_value,
                          "candidates": {t: {"objective": s[0], "eligible": s[1]} for t, s in scored.items()},
                          "settings_after": incumbent})
        (out_dir / "rounds.json").write_text(json.dumps(round_log, indent=2))
        log.info("Round '%s' winner: %s (objective %.4f; before %.4f)", name, winner, scored[winner][0], inc_value)
        tracking.log_metrics({"tune/rounds_done": len(round_log), f"tune/{name}/winner_objective": scored[winner][0],
                              f"tune/{name}/incumbent_objective": inc_value})

    (out_dir / "tuned_overrides.yaml").write_text(yaml.safe_dump(incumbent, sort_keys=True))
    tuned = write_tuned_config(config_path, grid_path, incumbent, grid)
    log.info("Tuning complete. Winning overrides: %s", incumbent)
    log.info("Tuned config written to %s — next: calibrate.py and run_experiment.py with it.", tuned)


def _add_arguments(parser) -> None:
    parser.add_argument("--grid", default=None,
                        help=f"tuning grid file (default: the config's tuning_grid, else {DEFAULT_GRID})")
    parser.add_argument("--full-refresh", action="store_true",
                        help="delete the tuning checkpoints and start over (default: resume)")


if __name__ == "__main__":
    args = parse_step_args("Hyperparameter search on the validation split.", add_arguments=_add_arguments)
    try:
        main(args.config_file, args.grid, full_refresh=args.full_refresh)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
