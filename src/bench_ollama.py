"""
Ollama throughput benchmark — run by jobs/bench_ollama.sh once per
Ollama setting (OLLAMA_NUM_PARALLEL, OLLAMA_FLASH_ATTENTION), against an
already-running server started with that setting.

Three modes:
  --label L --parallel N   forecast the same validation patients with one LLM
                           condition (default full_pipeline) at N concurrent
                           requests; record wall time, projected time per 450
                           patients, fallbacks, filled variables and mean
                           per-patient sMAPE (a sanity check that the setting
                           doesn't change the forecasts' quality; at
                           temperature 0.2 and 64 patients a few % is noise).
  --probe VARIANT --num-ctx C
                           load one LLM variant (e.g. qwen2_5_32b, or
                           'default' for llm.model) with
                           num_ctx C, send one short request, and record from
                           Ollama's /api/ps how much of the model sits in VRAM
                           (100% = fits; less = spilled to CPU, much slower).
  --summary                print every result recorded so far, with speed-up
                           relative to the first row (the baseline setting).

Patients come from the TUNING subset of the validation split (the same
seeded subset as src/tune.py); the test split is never read. Nothing is
checkpointed — this measures speed, it isn't part of the experiment.

Usage:
    python src/bench_ollama.py --config-file config/config_alliance_lean.yaml --label np4_fa0 --parallel 4
    python src/bench_ollama.py --config-file config/config_alliance_lean.yaml --probe default --num-ctx 12288 --label np8_fa1
    python src/bench_ollama.py --config-file config/config_alliance_lean.yaml --summary
Writes (appends):
    <results_dir>/ollama_bench/bench-<SLURM_JOB_ID>.jsonl
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from common import cfg_for_llm_variant, ensure_work_dirs, get_logger, load_config, parse_step_args
from evaluate_results import per_patient_smape
from harmonization_agent import build_tensors
from llm_client import LocalLLM
from pipeline import fit_models, run_condition
from run_experiment import load_cohort_and_panel, split_tensors
from tune import DEFAULT_GRID, config_for_conditions, load_grid, validation_subsets

log = get_logger("bench_ollama")

PROJECTED_PATIENTS = 450        # one condition on the test split


def results_path(cfg: dict) -> Path:
    out = Path(cfg["paths"]["results_dir"]) / "ollama_bench" / f"bench-{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def append(cfg: dict, row: dict) -> None:
    row = {**row,
           "num_parallel_env": os.environ.get("OLLAMA_NUM_PARALLEL"),
           "flash_attention_env": os.environ.get("OLLAMA_FLASH_ATTENTION")}
    with open(results_path(cfg), "a") as f:
        f.write(json.dumps(row) + "\n")
    log.info("Recorded: %s", row)


def throughput(cfg: dict, label: str, parallel: int, condition: str, n_patients: int) -> None:
    cfg = config_for_conditions(cfg, [condition])
    cfg["performance"]["llm_max_concurrent_requests"] = parallel
    cfg["performance"]["checkpoint_batch_size"] = n_patients     # one batch: full concurrency throughout
    grid = load_grid(DEFAULT_GRID)

    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    tune_ids, _ = validation_subsets(splits["val"], grid["n_tune_patients"], grid["seed"])
    ids = tune_ids[:n_patients]
    patients = {sid: splits["val"][sid] for sid in ids}
    fitted = fit_models(cfg, splits["train"])

    # Load the model before timing, so the numbers are per-patient work only.
    llm = fitted["forecaster"].llm
    t0 = time.time()
    llm.generate('Reply with the JSON object {"ok": true}.', json_mode=True)
    load_s = time.time() - t0
    log.info("Model %s loaded and answered in %.0fs.", llm.model, load_s)

    t0 = time.time()
    result = run_condition(condition, cfg, splits["train"], patients, fitted)
    wall = time.time() - t0
    pp = per_patient_smape(result["y_true"], result["y_pred"])
    n = len(ids)
    append(cfg, {
        "mode": "throughput", "label": label, "condition": condition, "model": llm.model,
        "parallel": parallel, "n_patients": n, "load_s": round(load_s, 1),
        "wall_s": round(wall, 1), "s_per_patient": round(wall / n, 2),
        f"projected_min_per_{PROJECTED_PATIENTS}": round(wall / n * PROJECTED_PATIENTS / 60, 1),
        "fallbacks": result["llm_fallbacks"], "filled": int(np.sum(result["llm_filled"])),
        "smape_mean": round(float(np.nanmean(pp)), 4),
    })


def probe(cfg: dict, label: str, variant: str, num_ctx: int) -> None:
    vcfg = copy.deepcopy(cfg_for_llm_variant(cfg, None if variant == "default" else variant))
    vcfg["llm"]["num_ctx"] = num_ctx
    llm = LocalLLM(vcfg)
    t0 = time.time()
    error = None
    try:
        llm.generate('Reply with the JSON object {"ok": true}.', json_mode=True)
    except Exception as e:  # noqa: BLE001 - a model that can't load is the result being measured
        error = str(e)[:300]
    load_s = time.time() - t0
    ps = llm._session.get(f"{llm.host}/api/ps", timeout=10).json().get("models", [])
    entry = next((m for m in ps if m.get("name", "").startswith(llm.model) or m.get("model", "").startswith(llm.model)), None)
    size, vram = (entry or {}).get("size", 0), (entry or {}).get("size_vram", 0)
    append(cfg, {
        "mode": "probe", "label": label, "variant": variant, "model": llm.model, "num_ctx": num_ctx,
        "load_s": round(load_s, 1), "size_gb": round(size / 1e9, 1), "vram_gb": round(vram / 1e9, 1),
        "on_gpu_pct": round(100 * vram / size, 1) if size else None, "error": error,
    })


def summary(cfg: dict) -> None:
    rows = []
    for path in sorted(results_path(cfg).parent.glob("bench-*.jsonl")):
        rows += [json.loads(line) | {"file": path.name} for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        print("No benchmark results yet.")
        return
    key = f"projected_min_per_{PROJECTED_PATIENTS}"
    for file in dict.fromkeys(r["file"] for r in rows):
        group = [r for r in rows if r["file"] == file]
        print(f"\n== {file} ==")
        tp = [r for r in group if r["mode"] == "throughput"]
        if tp:
            base = tp[0]
            print(f"{'setting':10s} {'parallel':>8s} {'FA':>3s} {'s/patient':>9s} {'min/450':>8s} {'speed-up':>8s} "
                  f"{'fallbacks':>9s} {'filled':>6s} {'sMAPE':>7s}")
            for r in tp:
                print(f"{r['label']:10s} {r['parallel']:>8d} {str(r['flash_attention_env']):>3s} {r['s_per_patient']:>9.2f} "
                      f"{r[key]:>8.1f} {base['wall_s'] / r['wall_s']:>7.2f}x {r['fallbacks']:>4d}/{r['n_patients']:<4d} "
                      f"{r['filled']:>6d} {r['smape_mean']:>7.4f}")
        for r in (r for r in group if r["mode"] == "probe"):
            fit = ("ERROR " + r["error"]) if r["error"] else (
                "fits on GPU" if (r["on_gpu_pct"] or 0) >= 99.9 else f"only {r['on_gpu_pct']}% on GPU — spills to CPU")
            print(f"probe {r['label']}: {r['variant']} num_ctx={r['num_ctx']} parallel={r['num_parallel_env']} "
                  f"FA={r['flash_attention_env']}: {r['size_gb']} GB, {fit}")


def main() -> None:
    def _add(p):
        p.add_argument("--label", default="")
        p.add_argument("--parallel", type=int, default=None)
        p.add_argument("--condition", default="full_pipeline")
        p.add_argument("--n-patients", type=int, default=64)
        p.add_argument("--probe", metavar="VARIANT", default=None)
        p.add_argument("--num-ctx", type=int, default=None)
        p.add_argument("--summary", action="store_true")
    args = parse_step_args(__doc__.strip().splitlines()[0], add_arguments=_add)
    cfg = load_config(args.config_file)
    ensure_work_dirs(cfg)
    if args.summary:
        summary(cfg)
    elif args.probe:
        probe(cfg, args.label, args.probe, args.num_ctx or cfg["llm"]["num_ctx"])
    else:
        if args.parallel is None:
            raise SystemExit("--parallel is required for a throughput run")
        throughput(cfg, args.label, args.parallel, args.condition, args.n_patients)


if __name__ == "__main__":
    main()
