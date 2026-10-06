"""
Output probe — what one LLM variant actually generates for the forecasting
prompt, with and without the JSON-schema constraint. Run by
jobs/probe_output.sh, against an already-running Ollama server.

Written for Baichuan-M2 in calibrate job 1032638 (Trillium), where every
forecast hit max_tokens=2048 under the schema introduced to fix Qwen's
30-31-value lists; the log showed the token count but not the text.

For each of a few patients (the TUNING subset of the validation split, as in
src/tune.py; the test split is never read) it sends the single-model
forecasting prompt — the exact prompt ForecastingAgent.build_prompt builds
from the config — once per output mode:
  schema_nothink  the schema with think=false (Ollama's reasoning switch off)
  schema          format = the forecast JSON schema (what the pipeline sends now)
  json            format = "json" (what it sent before)
one request at a time, with a larger token budget (--max-tokens) and timeout
than the experiment, so each answer can end on its own and its full length
is visible. Per call it records why generation stopped, tokens generated,
whether the answer parses with every variable at the horizon length, the
longest run of decimal digits in a number, and the first and last 300
characters of the answer and of Ollama's separate `thinking` (reasoning)
field.

Nothing is checkpointed and the pipeline's metrics aren't touched.

Usage:
    python src/probe_output.py --config-file config/config_alliance_lean_tuned.yaml --variant baichuan_m2
    python src/probe_output.py --config-file <cfg> --variant default --n-patients 4 --max-tokens 4096
Writes (one line per call, full output included):
    <results_dir>/probe_output/probe-<SLURM_JOB_ID>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from common import cfg_for_llm_variant, ensure_work_dirs, get_logger, load_config
from forecasting_agent import SYSTEM_PROMPT, ForecastingAgent
from harmonization_agent import reference_stats
from harmonization_agent import build_tensors
from llm_client import LocalLLM, extract_json_block
from run_experiment import load_cohort_and_panel, split_tensors
from tune import DEFAULT_GRID, load_grid, validation_subsets

log = get_logger("probe_output")

# schema_nothink first: the quickest answer (reasoning off). schema / json run
# with Ollama's default think; a reasoning model (Baichuan-M2, job 1033722)
# can spend the whole budget in the `thinking` field, recorded separately.
MODES = ("schema_nothink", "schema", "json")


def call(llm: LocalLLM, prompt: str, fmt, max_tokens: int, timeout: int, system: str = SYSTEM_PROMPT,
         think: bool | None = None) -> dict:
    """One /api/generate call, as LocalLLM.generate sends it but returning
    Ollama's whole response body (done_reason, eval_count, ...)."""
    payload = {
        "model": llm.model, "prompt": prompt, "system": system, "stream": False,
        "format": fmt,
        "options": {"temperature": llm.temperature, "num_predict": max_tokens, "num_ctx": llm.num_ctx},
    }
    if think is not None:
        payload["think"] = think
    t0 = time.time()
    resp = llm._session.post(f"{llm.host}/api/generate", json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    body["wall_s"] = round(time.time() - t0, 1)
    return body


def describe(text: str, variables: list[str], horizon: int) -> dict:
    """Shape of one answer: parses?, values per variable, longest decimals."""
    out = {"parses": False, "lengths": None, "all_lengths_ok": False,
           "max_decimals": max((len(d) for d in re.findall(r"\d\.(\d+)", text)), default=0)}
    try:
        parsed = extract_json_block(text)
        fc = parsed.get("forecast", {}) if isinstance(parsed, dict) else {}
        out["parses"] = True
        out["lengths"] = {v: (len(fc[v]) if isinstance(fc.get(v), list) else None) for v in variables}
        out["all_lengths_ok"] = all(n == horizon for n in out["lengths"].values())
    except Exception as e:  # noqa: BLE001 - an unparseable answer is a result, not an error
        out["parse_error"] = str(e)[:200]
    return out


def main(config_path: str, variant: str, n_patients: int, max_tokens: int, timeout: int) -> None:
    cfg = load_config(config_path)
    ensure_work_dirs(cfg)
    vcfg = cfg_for_llm_variant(cfg, None if variant == "default" else variant)
    llm = LocalLLM(vcfg)
    grid = load_grid(DEFAULT_GRID)
    cohort, panel_long = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel_long), cohort)
    variables = [v["name"] for v in cfg["variables"]]
    agent = ForecastingAgent(llm, vcfg, reference_stats(splits["train"], variables))
    horizon = agent.horizon_hours
    tune_ids, _ = validation_subsets(splits["val"], grid["n_tune_patients"], grid["seed"])
    ids = tune_ids[:n_patients]

    out_path = Path(cfg["paths"]["results_dir"]) / "probe_output" / f"probe-{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the model first (as run_condition now does), so load time and
    # generation are measured apart. Job 1033530 timed out on a short JSON
    # warm-up at the experiment's 180 s while the model was still loading.
    llm.load()

    # A trivial JSON request (bench_ollama.py's warm-up): does even this end
    # on its own, in JSON mode and under a schema?
    rows = []
    ok_prompt = 'Reply with the JSON object {"ok": true}.'
    for mode, fmt in (("ok_json", "json"), ("ok_schema", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                          "required": ["ok"]})):
        row = {"variant": variant, "model": llm.model, "stay_id": None, "mode": mode, "max_tokens": 256}
        try:
            body = call(llm, ok_prompt, fmt, 256, timeout, system="")
            text = body.get("response", "")
            row.update({"done_reason": body.get("done_reason"), "tokens": body.get("eval_count"),
                        "wall_s": body["wall_s"], "chars": len(text), "start": text[:300], "end": text[-300:],
                        "response": text})
        except Exception as e:  # noqa: BLE001 - a timeout is a result too
            row.update({"done_reason": "error", "error": str(e)[:300]})
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log.info("warm-up %-9s: %s, %s tokens, %ss\n    starts: %r\n    ends:   %r", mode, row["done_reason"],
                 row.get("tokens"), row.get("wall_s"), row.get("start", row.get("error")), row.get("end", ""))

    log.info("Probing %d patients x %s, max_tokens=%d, timeout=%ds; writing %s",
             len(ids), "/".join(MODES), max_tokens, timeout, out_path)
    for sid in ids:
        prompt = agent.build_prompt(splits["val"][sid]["obs"])
        for mode in MODES:
            fmt = "json" if mode == "json" else agent.schema
            think = False if mode == "schema_nothink" else None
            row = {"variant": variant, "model": llm.model, "stay_id": int(sid), "mode": mode,
                   "max_tokens": max_tokens}
            try:
                body = call(llm, prompt, fmt, max_tokens, timeout, system=agent.system_prompt, think=think)
                text = body.get("response", "")
                thinking = body.get("thinking") or ""
                row.update({"done_reason": body.get("done_reason"), "tokens": body.get("eval_count"),
                            "wall_s": body["wall_s"], "chars": len(text),
                            **describe(text, variables, horizon),
                            "start": text[:300], "end": text[-300:], "response": text,
                            "thinking_chars": len(thinking), "thinking_start": thinking[:300],
                            "thinking_end": thinking[-300:], "thinking": thinking})
            except Exception as e:  # noqa: BLE001 - a timeout is a result too
                row.update({"done_reason": "error", "error": str(e)[:300]})
            rows.append(row)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            log.info("stay %s %-14s: %s, %s tokens, %ss, parses=%s, lengths ok=%s, max decimals=%s, "
                     "reasoning %s chars\n    starts: %r\n    ends:   %r\n"
                     "    reasoning starts: %r\n    reasoning ends:   %r",
                     sid, mode, row["done_reason"], row.get("tokens"), row.get("wall_s"), row.get("parses"),
                     row.get("all_lengths_ok"), row.get("max_decimals"), row.get("thinking_chars"),
                     row.get("start", row.get("error")), row.get("end", ""),
                     row.get("thinking_start", ""), row.get("thinking_end", ""))

    print(f"\n== {variant} ({llm.model}), {len(ids)} patients, horizon {horizon} h, max_tokens {max_tokens} ==")
    print(f"{'mode':14s} {'stopped':>8s} {'length':>7s} {'tokens (median/max)':>20s} {'usable':>7s} {'max decimals':>12s}")
    for mode in MODES:
        rs = [r for r in rows if r["mode"] == mode]
        toks = sorted(r["tokens"] for r in rs if r.get("tokens") is not None)
        med = toks[len(toks) // 2] if toks else "-"
        print(f"{mode:14s} {sum(r['done_reason'] == 'stop' for r in rs):>4d}/{len(rs):<3d} "
              f"{sum(r['done_reason'] == 'length' for r in rs):>3d}/{len(rs):<3d} "
              f"{str(med) + '/' + str(toks[-1] if toks else '-'):>20s} "
              f"{sum(bool(r.get('all_lengths_ok')) for r in rs):>3d}/{len(rs):<3d} "
              f"{max((r.get('max_decimals') or 0) for r in rs):>12d}")
    print("stopped = ended on its own; length = hit max_tokens; usable = parses with every variable "
          f"at {horizon} values. Full outputs: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-file", default="config/config_alliance_lean.yaml")
    parser.add_argument("--variant", default="baichuan_m2",
                        help="an llm.variants name, or 'default' for llm.model")
    parser.add_argument("--n-patients", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="token budget per call (the experiment uses llm.max_tokens)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per call")
    args = parser.parse_args()
    main(args.config_file, args.variant, args.n_patients, args.max_tokens, args.timeout)
