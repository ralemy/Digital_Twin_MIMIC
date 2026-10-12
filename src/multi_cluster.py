"""
One run on several clusters: the target cluster holds the run, helpers take
a share of calibration and the test stage. Runs on a HUB, a login node with an
open SSH connection to every cluster (config/multi_cluster.yaml); it submits
the jobs through each cluster's jobs/submit.sh, polls Slurm (one sacct query
per cluster per check), and copies finished work between clusters through the
hub. Without it, jobs/run_all.sh runs everything on one cluster, as before.

Assumed on every cluster (checked, not installed, before anything is
submitted): the repo is set up with jobs/setup_bash.sh, so the profile
(~/.config/dt_profile.yml, read through jobs/load_profile.sh as the jobs read
it) gives its locations; the Ollama models the cluster's share uses are in
$OLLAMA_MODELS (the target: every model of the run, since it computes any
condition a helper does not deliver); and, on the target, the repo's mimic-iv
link resolves to MIMIC-IV. Helpers never read MIMIC-IV: they get the target's
extracted cohort. A missing item is reported at every check until it is fixed.

Sequence (each step starts only when the previous one is complete, whatever
order the clusters finish in):
  1. prepare: on the target, jobs/run_all.sh <scope> --until tune (models,
     resolve, extract, tune).
  2. inputs: the target's cohort.parquet, panel_long.parquet, cache/ and
     tuned config are copied to each helper; every condition's calibration
     fingerprint is computed on each cluster and must match.
  3. calibrate: each cluster runs step 3c with --conditions <its share>; each
     condition finished on a helper is copied into the target's checkpoints
     (verified: batch count and fingerprint).
  4. calibrate-assemble: step 3c of all conditions on the target, which loads
     every condition from the checkpoints (conditions a helper did not deliver
     are computed here) and writes the complete calibration.json.
  5. calibration.json goes to the helpers; the test-stage fingerprints (which
     include the fitted post-processing) must match.
  6. run, run-assemble: as 3-4 with step 3 (each condition's checkpoint and
     <condition>_raw.npz / _summary.txt are copied).
  7. evaluate on the target; the helpers' job logs and a record of who ran
     what (multi_cluster.json in the work dir) are copied there; pack_run.sh.

The baselines run on the target; LLM conditions are shared out by estimated
GPU time (model_weights, speed) unless the config fixes the assignment. A
helper share that fails is resubmitted (max_retries); after that, or when the
helper stays unreachable for give_up_hours, its unfinished conditions are left
to the target's assemble step: slower, never wrong. State is kept in
multi_runs/<run>/state.json (log: multi.log), so the orchestrator can be
stopped and started again at any time; the jobs keep running meanwhile.

Usage (on the hub, from the repository base):
    python src/multi_cluster.py plan   <lean|full> --run-name <run> [--target fir --helpers trillium,rorqual] [--config <base config>] [--multi config/multi_cluster.yaml]
    python src/multi_cluster.py start  <lean|full> --run-name <run> [--target fir --helpers trillium,rorqual] [--config <base config>] [--multi ...]
    python src/multi_cluster.py status <run>
    python src/multi_cluster.py stop   <run>
(run instead of start stays in the foreground.) --target and --helpers
override the multi-cluster file's target and helpers (each must be under its
clusters:); a run keeps the ones it started with.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
STANDARD_CONFIG = {"lean": "config/config_alliance_lean.yaml", "full": "config/config_alliance_full.yaml"}
JOB_SCRIPT = {"calibrate": "jobs/step3c_calibrate.sh", "run": "jobs/step3_run_experiment.sh",
              "evaluate": "jobs/step4_evaluate_results.sh"}
CKPT_DIR = {"calibrate": "calibration", "run": "run_experiment"}
# Every remote command: the cluster's profile (locations, account, modules,
# DT_LOGS_ROOT) as the jobs and jobs/run_all.sh see it, from the repo base.
PRELUDE = ('source ~/.bashrc >/dev/null 2>&1; cd "$DT_REPO" 2>/dev/null || { echo "DT_REPO not set" >&2; exit 97; }; '
           'source jobs/load_profile.sh >/dev/null 2>&1 && load_profile >/dev/null 2>&1; cd "$DT_REPO"; ')
PYENV = 'module load $DT_MODULES >/dev/null 2>&1; source .venv/bin/activate; '
ENDED = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
         "BOOT_FAIL", "DEADLINE")
q = shlex.quote


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

class Cluster:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.host = spec["host"]
        self.local = self.host == "local"
        self.control_path = os.path.expanduser(spec.get("control_path", "")) if not self.local else None
        self.max_job_hours = float(spec.get("max_job_hours", 8))
        self.speed = float(spec.get("speed", 1.0))

    def _argv(self, script: str) -> list[str]:
        cmd = PRELUDE + script
        if self.local:
            return ["bash", "-c", cmd]
        return ["ssh", "-o", f"ControlPath={self.control_path}", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                self.host, "bash -c " + q(cmd)]

    def run(self, script: str, timeout: int = 300) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self._argv(script), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(e.cmd, 124, "", f"timed out after {timeout} s")

    def popen(self, script: str, **kw) -> subprocess.Popen:
        return subprocess.Popen(self._argv(script), **kw)

    def alive(self) -> bool:
        if self.local:
            return True
        r = subprocess.run(["ssh", "-o", f"ControlPath={self.control_path}", "-O", "check", self.host],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0


# ---------------------------------------------------------------------------
# Plan: who runs what
# ---------------------------------------------------------------------------

def condition_weight(cfg: dict, condition: str, multi: dict) -> float:
    from common import LLM_CONDITIONS, cfg_for_llm_variant, ollama_model_name, split_condition
    base, variant = split_condition(condition)
    if base not in LLM_CONDITIONS:
        return float(multi.get("baseline_weight", 0.15))
    llm = cfg_for_llm_variant(cfg, variant)["llm"]
    names = (ollama_model_name(llm) + " " + llm["model"]).lower()
    for rule in multi.get("model_weights") or [{"match": "", "weight": 1.0}]:
        if str(rule["match"]).lower() in names:
            return float(rule["weight"])
    return 1.0


def assign(cfg: dict, multi: dict) -> dict[str, list[str]]:
    """{cluster: [conditions]} in the config's condition order. Baselines on
    the target; LLM conditions longest-first to the cluster that would finish
    its share soonest (weight / speed), the target starting with its
    baselines and the assemble steps' share."""
    from common import LLM_CONDITIONS, split_condition
    target, helpers = multi["target"], list(multi.get("helpers") or [])
    clusters = [target] + helpers
    speed = {c: float(multi["clusters"][c].get("speed", 1.0)) for c in clusters}
    out = {c: [] for c in clusters}
    fixed = {c: list(v) for c, v in (multi.get("assignment") or {}).items()}
    placed = {x for v in fixed.values() for x in v}
    unknown = sorted(placed - set(cfg["conditions"]))
    if unknown:
        raise ValueError(f"assignment names conditions not in the config: {unknown}")
    for c, conds in fixed.items():
        if c not in out:
            raise ValueError(f"assignment names cluster '{c}', not the target or a helper")
        out[c].extend(conds)
    load = {c: sum(condition_weight(cfg, x, multi) for x in out[c]) / speed[c] for c in clusters}
    load[target] += 0.3        # the assemble, evaluate and pack steps
    free = [x for x in cfg["conditions"] if x not in placed]
    for x in free:
        if split_condition(x)[0] not in LLM_CONDITIONS:
            out[target].append(x)
            load[target] += condition_weight(cfg, x, multi) / speed[target]
    llm = sorted((x for x in free if split_condition(x)[0] in LLM_CONDITIONS),
                 key=lambda x: -condition_weight(cfg, x, multi))
    for x in llm:
        c = min(clusters, key=lambda k: (load[k] + condition_weight(cfg, x, multi) / speed[k], k != target))
        out[c].append(x)
        load[c] += condition_weight(cfg, x, multi) / speed[c]
    order = {x: i for i, x in enumerate(cfg["conditions"])}
    return {c: sorted(v, key=order.get) for c, v in out.items()}


def share_minutes(cfg: dict, multi: dict, scope: str, stage: str, conditions: list[str], speed: float) -> float:
    per_unit = (multi.get("minutes_per_unit") or {}).get(scope, {}).get(stage, 120)
    return sum(condition_weight(cfg, x, multi) for x in conditions) * per_unit / speed


def chain_shape(minutes: float, max_job_hours: float) -> tuple[int, str]:
    """(number of chained jobs, time limit of each) for this much work, with
    30 % margin and 20 minutes of start-up per job."""
    limit = int(max_job_hours * 60)
    need = minutes * 1.3 + 20
    n = max(1, min(8, math.ceil(need / (limit - 20))))
    per = limit if n > 1 else max(60, min(limit, int(math.ceil(need / 15) * 15)))
    return n, f"{per // 60:02d}:{per % 60:02d}:00"


# ---------------------------------------------------------------------------
# Remote side: paths, fingerprints, completeness (run on each cluster)
# ---------------------------------------------------------------------------

def remote_info(config: str, stage: str | None, paths_only: bool) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from checkpoint import _hash, checkpoint_root, condition_fingerprint
    from common import load_config
    cfg = load_config(config)
    out = {"paths": {k: str(cfg["paths"][k]) for k in ("work_dir", "cache_dir", "results_dir")},
           "checkpoint_dir": str(checkpoint_root(cfg)), "conditions": cfg["conditions"]}
    if paths_only or stage is None:
        return out
    import math as _m
    from harmonization_agent import build_tensors
    from run_experiment import fitted_postprocess, load_cohort_and_panel, split_tensors
    from tune import config_grid, load_grid, validation_subsets
    cohort, panel = load_cohort_and_panel(cfg)
    splits = split_tensors(build_tensors(cfg, cohort, panel), cohort)
    train_ids = sorted(splits["train"])
    bs = max(1, int(cfg.get("performance", {}).get("checkpoint_batch_size", 32)))
    if stage == "calibrate":
        grid = load_grid(config_grid(cfg))
        n_tune = (cfg.get("tuned_from") or {}).get("n_tune_patients", grid["n_tune_patients"])
        seed = (cfg.get("tuned_from") or {}).get("seed", grid["seed"])
        _, ids = validation_subsets(splits["val"], n_tune, seed)
        fps = {c: condition_fingerprint(cfg, c, train_ids, ids) for c in cfg["conditions"]}
    else:
        ids = sorted(splits["test"])
        pp = fitted_postprocess(cfg, Path(cfg["paths"]["results_dir"]), train_ids)
        fps = {c: condition_fingerprint(cfg, c, train_ids, ids, postprocess=pp.get(c)) for c in cfg["conditions"]}
    out["n_batches"] = _m.ceil(len(ids) / bs)
    out["digests"] = {c: _hash(fp) for c, fp in fps.items()}
    return out


def manifest_path(models_dir: str, name: str) -> Path:
    """Where Ollama keeps a model's manifest: hf.co/<user>/<repo>:<tag> under
    manifests/hf.co/<user>/<repo>/<tag>, a library name (or an alias made
    with ollama cp) under manifests/registry.ollama.ai/library/<name>/<tag>."""
    repo, _, tag = name.partition(":")
    tag = tag or "latest"
    if repo.startswith("hf.co/"):
        return Path(models_dir) / "manifests" / repo / tag
    host_path = repo if "/" in repo else f"library/{repo}"
    return Path(models_dir) / "manifests" / "registry.ollama.ai" / host_path / tag


def remote_check(config: str, conditions: list[str], mimic: bool) -> dict:
    """Run on a cluster: are the models for these conditions present, and
    (mimic) does the repo's mimic-iv link resolve to MIMIC-IV?"""
    sys.path.insert(0, str(REPO / "src"))
    from common import llm_models_in_use, load_config, split_condition
    cfg = load_config(config)
    bases = {split_condition(c)[0] for c in conditions}
    sub_cfg = {**cfg, "conditions": list(conditions),
               "baselines": {**cfg["baselines"], "run_single_model_llm": "single_model_llm" in conditions}}
    models_dir = os.environ.get("OLLAMA_MODELS") or os.environ.get("DT_OLLAMA_MODELS") or ""
    models = llm_models_in_use(sub_cfg) if bases else []
    out = {"models_dir": models_dir, "missing_models": [m for m in models if not models_dir
                                                         or not manifest_path(models_dir, m).exists()]}
    if mimic:
        root = REPO / "mimic-iv"
        need = [("icu", "d_items"), ("icu", "chartevents"), ("hosp", "labevents"), ("hosp", "d_labitems")]
        out["mimic_missing"] = [f"{d}/{n}.csv[.gz]" for d, n in need
                                if not ((root / d / f"{n}.csv.gz").exists() or (root / d / f"{n}.csv").exists())]
        out["mimic_dir"] = str(root.resolve())
    return out


def finished_script(ckroot: str, results_dir: str, stage: str, expected: dict, n_batches: int,
                    conditions: list[str]) -> str:
    """Shell that prints each of these conditions whose checkpoint here is
    complete and has the expected fingerprint."""
    lines = []
    for c in conditions:
        d = f"{ckroot}/{CKPT_DIR[stage]}/conditions/{c}"
        dig = expected[c]
        if stage == "run":
            test = (f'[ -f {q(d + "/summary_rows.json")} ] && [ -f {q(results_dir + "/" + c + "_raw.npz")} ]')
        else:
            test = f'[ "$(ls {q(d)}/batch_*.npz 2>/dev/null | wc -l)" -ge {n_batches} ]'
        lines.append(f'{test} && grep -q {q(dig)} {q(d + "/fingerprint.json")} 2>/dev/null && echo {q(c)}')
    return "; ".join(lines) + "; true"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, run: str, state_dir: Path):
        self.run_name = run
        self.dir = state_dir
        self.state_path = state_dir / "state.json"
        self.log_path = state_dir / "multi.log"
        self.s = json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    # -- bookkeeping -------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
        if sys.stdout.isatty():
            print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=2))
        tmp.replace(self.state_path)

    def init(self, scope: str, base_config: str, multi_path: str, multi: dict) -> None:
        sys.path.insert(0, str(REPO / "src"))
        from common import load_config
        cfg = load_config(base_config)
        tuned = f"tuned_configs/{self.run_name}/{Path(base_config).stem}_tuned.yaml"
        self.s = {"run": self.run_name, "scope": scope, "base_config": base_config, "multi": multi,
                  "multi_path": multi_path, "tuned_config": tuned, "assignment": assign(cfg, multi),
                  "phase": "preflight", "env": {}, "expected": {}, "parts": {}, "assemble": {},
                  "started": datetime.now().isoformat(timespec="seconds"), "events": []}
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save()

    @property
    def multi(self) -> dict:
        return self.s["multi"]

    def cluster(self, name: str) -> Cluster:
        return Cluster(name, self.multi["clusters"][name])

    @property
    def target(self) -> str:
        return self.multi["target"]

    @property
    def helpers(self) -> list[str]:
        return list(self.multi.get("helpers") or [])

    def event(self, msg: str) -> None:
        self.s["events"].append({"time": datetime.now().isoformat(timespec="seconds"), "event": msg})
        self.log(msg)

    def cfg(self) -> dict:
        sys.path.insert(0, str(REPO / "src"))
        from common import load_config
        return load_config(self.s["base_config"])

    # -- remote helpers ----------------------------------------------------
    def sh(self, name: str, script: str, timeout: int = 300) -> subprocess.CompletedProcess:
        return self.cluster(name).run(script, timeout=timeout)

    def info(self, name: str, stage: str | None, paths_only: bool = False) -> dict | None:
        flag = " --paths-only" if paths_only else ""
        st = f" --stage {stage}" if stage else ""
        r = self.sh(name, PYENV + f"python src/multi_cluster.py remote-info {q(self.s['tuned_config'])}{st}{flag} 2>/dev/null",
                    timeout=1200)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            self.log(f"{name}: remote-info failed (rc {r.returncode}): {r.stderr.strip()[-300:]}")
            return None

    def submit(self, name: str, stage: str, conditions: list[str] | None, n: int, limit: str) -> list[str]:
        job = JOB_SCRIPT[stage]
        extra = f" --conditions {q(','.join(conditions))}" if conditions else ""
        ids, dep = [], ""
        for _ in range(n):
            r = self.sh(name, f'mkdir -p "$DT_LOGS_ROOT/{self.run_name}"; bash jobs/submit.sh --parsable --time={limit} '
                              f'--output="$DT_LOGS_ROOT/{self.run_name}/%x-%j.out"{dep} {job} {q(self.s["tuned_config"])}{extra}')
            m = re.search(r"(\d+)", r.stdout or "")
            if r.returncode != 0 or not m:
                self.log(f"{name}: submitting {stage} failed: {(r.stderr or r.stdout).strip()[-300:]}")
                break
            ids.append(m.group(1))
            dep = f" --dependency=afternotok:{m.group(1)}"
        if ids:
            self.event(f"{name}: {stage} submitted ({len(conditions) if conditions else 'all'} conditions), "
                       f"jobs {', '.join(ids)} x {limit}")
        return ids

    def job_states(self, name: str, ids: list[str]) -> dict[str, tuple[str, str]] | None:
        if not ids:
            return {}
        r = self.sh(name, f"sacct -X -n -P -j {','.join(ids)} -o JobID,State,ExitCode", timeout=120)
        if r.returncode != 0:
            return None
        out = {}
        for line in r.stdout.splitlines():
            p = line.strip().split("|")
            if len(p) >= 3 and p[0] in ids:
                out[p[0]] = (p[1].split()[0], p[2])
        for i in ids:            # not yet known to sacct right after submission
            out.setdefault(i, ("PENDING", ""))
        return out

    def copy(self, src: str, dst: str, src_parent: str, items: list[tuple[str, str]], dst_dirs: dict[str, str],
             finish: str) -> str:
        """tar `items` ([(parent key, name)]) on src, unpack into a temporary
        directory on dst and run `finish` there ($X is the directory)."""
        args = " ".join(f"-C {q(src_parent[k] if isinstance(src_parent, dict) else src_parent)} {q(n)}" for k, n in items)
        p1 = self.cluster(src).popen(f"tar cf - {args}", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        base = dst_dirs["base"]
        p2 = self.cluster(dst).popen(f'set -e; mkdir -p {q(base)}; X=$(mktemp -d {q(base)}/.incoming.XXXX); '
                                     f'tar xf - -C "$X"; {finish}; rm -rf "$X"',
                                     stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1.stdout.close()
        try:
            out, err = p2.communicate(timeout=3600)
            p1.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p1.kill(); p2.kill()
            return "timeout"
        if p2.returncode != 0 or p1.returncode != 0:
            return f"failed: {err.decode(errors='replace').strip()[-300:]}"
        return out.decode().strip() or "ok"

    # -- phases ------------------------------------------------------------
    def preflight(self) -> bool:
        hub_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        for name in [self.target] + self.helpers:
            c = self.cluster(name)
            if not c.alive():
                self.log(f"{name}: no SSH connection ({c.control_path}) — open it (Duo) and the next check carries on")
                return False
            r = self.sh(name, 'echo "$DT_REPO|$DT_RESULTS_DIR|$DT_LOGS_ROOT|$(git rev-parse HEAD)|'
                              '$(git status --porcelain --untracked-files=no | wc -l)|'
                              '$(squeue -h -u $USER -o %j 2>/dev/null | grep -c mimic-twin)"')
            parts = r.stdout.strip().split("|")
            if r.returncode != 0 or len(parts) != 6:
                self.log(f"{name}: could not read its profile: {r.stderr.strip()[-200:]}")
                return False
            repo, results, logs, head, dirty, jobs = parts
            if head != hub_head:
                if self.multi.get("auto_pull", False) and dirty == "0" and jobs == "0":
                    p = self.sh(name, "git pull -q && git rev-parse HEAD")
                    head = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else head
                    self.event(f"{name}: git pull -> {head[:7]}")
                if head != hub_head:
                    self.log(f"{name}: checkout {head[:7]} differs from the hub's {hub_head[:7]} "
                             f"(local changes: {dirty}, queued/running project jobs: {jobs}) — pull there, then this continues")
                    return False
            if not results or not logs:
                self.log(f"{name}: profile gives no DT_RESULTS_DIR/DT_LOGS_ROOT — run jobs/setup_bash.sh there")
                return False
            self.s["env"][name] = {"repo": repo, "results": results, "logs": logs, "head": head}
        problems = []
        for name in [self.target] + self.helpers:
            # the target may have to compute any condition: it needs every model
            conds = self.cfg()["conditions"] if name == self.target else (self.s["assignment"].get(name) or [])
            if not conds and name != self.target:
                continue
            flag = " --mimic" if name == self.target else ""
            r = self.sh(name, PYENV + f"python src/multi_cluster.py remote-check {q(self.s['base_config'])} "
                                      f"--conditions {q(','.join(conds))}{flag} 2>/dev/null", timeout=600)
            try:
                chk = json.loads(r.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                problems.append(f"{name}: could not check its models (rc {r.returncode}) {r.stderr.strip()[-200:]}")
                continue
            if chk["missing_models"]:
                problems.append(f"{name}: models missing from {chk['models_dir'] or '$OLLAMA_MODELS (not set)'}: "
                                f"{chk['missing_models']} (jobs/prep2_download_models.sh there)")
            if chk.get("mimic_missing"):
                problems.append(f"{name}: the repo's mimic-iv link ({chk['mimic_dir']}) lacks {chk['mimic_missing']} "
                                f"— point it at MIMIC-IV (paths.mimic_dir in the profile, then jobs/setup_bash.sh)")
        if problems:
            for msg in problems:
                self.log(msg)
            return False
        self.event(f"preflight: all clusters at {hub_head[:7]}; assignment " +
                   "; ".join(f"{c}: {len(v)}" for c, v in self.s["assignment"].items()))
        return True

    def start_driver(self, name: str, until: str) -> None:
        sc = self.s["scope"]
        self.sh(name, f"timeout 60 bash jobs/run_all.sh {sc} --config {q(self.s['base_config'])} --run-name "
                      f"{q(self.run_name)} --until {until} --unattended >/dev/null 2>&1; true", timeout=180)

    def driver_status(self, name: str, stage: str) -> tuple[str | None, bool]:
        sc = self.s["scope"]
        r = self.sh(name, f'st=run_all/{q(self.run_name)}/{sc}.state; hb=run_all/{q(self.run_name)}/{sc}.heartbeat; '
                          f'awk \'$1=="{stage}"{{print $2}}\' "$st" 2>/dev/null | tail -1; '
                          f'[ -f "$hb" ] && [ $(( $(date +%s) - $(stat -c %Y "$hb") )) -lt 600 ] && echo ALIVE || echo DEAD')
        lines = r.stdout.split()
        if r.returncode != 0 or not lines:
            return None, False
        return (lines[0] if len(lines) > 1 else None), lines[-1] == "ALIVE"

    def prepare(self) -> bool:
        prep = self.s.setdefault("prepare", {})
        todo = {self.target: "tune"}       # helpers need nothing before step 2 (checked in preflight)
        done = True
        for name, until in todo.items():
            p = prep.setdefault(name, {"started": False, "restarts": 0, "done": False})
            if p["done"]:
                continue
            if not p["started"]:
                self.start_driver(name, until)
                p["started"] = True
                self.event(f"{name}: run_all.sh --until {until} started")
                done = False
                continue
            status, alive = self.driver_status(name, until)
            if status == "done":
                p["done"] = True
                self.event(f"{name}: {until} done")
                continue
            done = False
            if not alive and status != "done":
                if p["restarts"] < self.multi.get("max_retries", 3):
                    p["restarts"] += 1
                    self.start_driver(name, until)
                    self.event(f"{name}: run_all.sh driver not alive ({until}: {status}) — restarted ({p['restarts']})")
                else:
                    self.log(f"{name}: run_all.sh driver keeps stopping before {until} is done — see its run_all log")
        return done

    def sync_inputs(self) -> bool:
        t = self.target
        tinfo = self.info(t, "calibrate")
        if not tinfo:
            return False
        self.s["paths"] = {t: tinfo["paths"] | {"checkpoint_dir": tinfo["checkpoint_dir"]}}
        self.s["expected"]["calibrate"] = {"digests": tinfo["digests"], "n_batches": tinfo["n_batches"]}
        tenv = self.s["env"][t]
        for h in self.helpers:
            # the tuned config first (it names the work dir), then the inputs
            res = self.copy(t, h, tenv["repo"], [("", f"tuned_configs/{self.run_name}")],
                            {"base": self.s["env"][h]["repo"] + "/tuned_configs"},
                            f'mkdir -p {q(self.s["env"][h]["repo"])}/tuned_configs/{q(self.run_name)}; '
                            f'cp -p "$X"/tuned_configs/{q(self.run_name)}/* {q(self.s["env"][h]["repo"])}/tuned_configs/{q(self.run_name)}/')
            if not res.startswith("ok"):
                self.log(f"{h}: copying the tuned config failed ({res})")
                return False
            hp = self.info(h, None, paths_only=True)
            if not hp:
                return False
            work = tinfo["paths"]["work_dir"]
            res = self.copy(t, h, work, [("", "cohort.parquet"), ("", "panel_long.parquet"), ("", "cache")],
                            {"base": hp["paths"]["work_dir"]},
                            f'cd "$X" && cp -rp cohort.parquet panel_long.parquet cache {q(hp["paths"]["work_dir"])}/')
            if not res.startswith("ok"):
                self.log(f"{h}: copying the cohort failed ({res})")
                return False
            hinfo = self.info(h, "calibrate")
            if not hinfo:
                return False
            self.s["paths"][h] = hinfo["paths"] | {"checkpoint_dir": hinfo["checkpoint_dir"]}
            bad = [c for c in self.s["assignment"][h] if hinfo["digests"].get(c) != tinfo["digests"].get(c)]
            if bad:
                self.s["phase"] = "error"
                self.event(f"{h}: calibration fingerprints differ from {t}'s for {bad} — stopped; check code and config")
                return False
            self.event(f"{h}: inputs copied; calibration fingerprints match")
        return True

    def sync_calibration(self) -> bool:
        t = self.target
        tr = self.s["paths"][t]["results_dir"]
        tinfo = self.info(t, "run")
        if not tinfo:
            return False
        missing = [c for c in tinfo["conditions"] if c not in tinfo["digests"]]
        if missing:
            self.log(f"{t}: no run fingerprints for {missing}")
            return False
        self.s["expected"]["run"] = {"digests": tinfo["digests"], "n_batches": tinfo["n_batches"]}
        for h in self.helpers:
            hr = self.s["paths"][h]["results_dir"]
            res = self.copy(t, h, tr, [("", "calibration.json"), ("", "calibration_smape.npz")], {"base": hr},
                            f'cp -p "$X"/calibration.json "$X"/calibration_smape.npz {q(hr)}/')
            if not res.startswith("ok"):
                self.log(f"{h}: copying calibration.json failed ({res})")
                return False
            hinfo = self.info(h, "run")
            if not hinfo:
                return False
            bad = [c for c in self.s["assignment"][h] if hinfo["digests"].get(c) != tinfo["digests"].get(c)]
            if bad:
                self.s["phase"] = "error"
                self.event(f"{h}: test-stage fingerprints differ from {t}'s for {bad} — stopped")
                return False
            self.event(f"{h}: calibration.json copied; test-stage fingerprints match")
        return True

    def stage_parts(self, stage: str) -> bool:
        """Submit, watch and collect the shares of a split stage. True when
        the target's share is done and every helper's is delivered or given up."""
        cfg = self.cfg()
        parts = self.s["parts"].setdefault(stage, {})
        exp = self.s["expected"][stage]
        t = self.target
        all_done = True
        for name in [t] + self.helpers:
            conds = self.s["assignment"].get(name) or []
            p = parts.setdefault(name, {"jobs": [], "status": "new", "retries": 0, "copied": [],
                                        "last_seen": time.time()})
            if not conds or p["status"] in ("done", "abandoned"):
                if not conds:
                    p["status"] = "done"
                continue
            cl = self.cluster(name)
            if not cl.alive():
                all_done = False
                if name != t and time.time() - p["last_seen"] > 3600 * float(self.multi.get("give_up_hours", 12)):
                    p["status"] = "abandoned"
                    self.event(f"{name}: unreachable for {self.multi.get('give_up_hours', 12)} h — its unfinished "
                               f"{stage} conditions are left to {t}")
                continue
            p["last_seen"] = time.time()
            if p["status"] == "new":
                todo = conds if name == t else [c for c in conds if c not in p["copied"]]
                n, limit = chain_shape(share_minutes(cfg, self.multi, self.s["scope"], stage, todo, cl.speed),
                                       cl.max_job_hours)
                p["jobs"] = self.submit(name, stage, todo, n, limit)
                p["status"] = "running" if p["jobs"] else "new"
                all_done = False
                continue
            # collect finished helper conditions as they come
            if name != t:
                self.collect(stage, name, conds, p)
            states = self.job_states(name, p["jobs"])
            if states is None:
                all_done = False
                continue
            completed = any(s == "COMPLETED" and e.startswith("0:") for s, e in states.values())
            ended = all(s in ENDED or s.startswith("CANCELLED") for s, _ in states.values())
            if name != t and set(p["copied"]) >= set(conds):
                p["status"] = "done"
                self.cancel_waiting(name, states)
                self.event(f"{name}: {stage} share delivered ({len(conds)} conditions)")
                continue
            if name == t and completed:
                p["status"] = "done"
                self.cancel_waiting(name, states)
                self.event(f"{name}: {stage} share done")
                continue
            if ended:
                if p["retries"] < int(self.multi.get("max_retries", 3)):
                    p["retries"] += 1
                    p["status"] = "new"
                    self.event(f"{name}: {stage} jobs ended ({', '.join(f'{j} {s}' for j, (s, _) in states.items())}) "
                               f"without finishing — resubmitting ({p['retries']})")
                elif name != t:
                    p["status"] = "abandoned"
                    self.event(f"{name}: {stage} retries used up — its unfinished conditions are left to {t}")
                else:
                    self.s["phase"] = "error"
                    self.event(f"{t}: its own {stage} share keeps failing — stopped; see its job logs")
                    return False
            all_done = False
        return all_done

    def cancel_waiting(self, name: str, states: dict) -> None:
        """Chained continuation jobs still queued once their work is done."""
        waiting = [j for j, (s, _) in states.items() if s in ("PENDING", "REQUEUED")]
        if waiting:
            self.sh(name, f"scancel {' '.join(waiting)}")

    def collect(self, stage: str, helper: str, conds: list[str], p: dict) -> None:
        exp = self.s["expected"][stage]
        hp, tp = self.s["paths"][helper], self.s["paths"][self.target]
        left = [c for c in conds if c not in p["copied"]]
        if not left:
            return
        r = self.sh(helper, finished_script(hp["checkpoint_dir"], hp["results_dir"], stage, exp["digests"],
                                            exp["n_batches"], left), timeout=120)
        for c in r.stdout.split():
            if c not in left:
                continue
            ck_src = f"{hp['checkpoint_dir']}/{CKPT_DIR[stage]}/conditions"
            ck_dst = f"{tp['checkpoint_dir']}/{CKPT_DIR[stage]}/conditions"
            items = [("ck", c)]
            parents = {"ck": ck_src}
            finish = []
            if stage == "run":
                items += [("res", f"{c}_raw.npz"), ("res", f"{c}_summary.txt")]
                parents["res"] = hp["results_dir"]
                finish.append(f'mkdir -p {q(tp["results_dir"])}; '
                              f'cp -p "$X"/{q(c + "_raw.npz")} "$X"/{q(c + "_summary.txt")} {q(tp["results_dir"])}/')
            d = f"{ck_dst}/{c}"
            finish = [f'mkdir -p {q(ck_dst)}; '
                      f'if ls {q(d)}/batch_*.npz >/dev/null 2>&1; then rm -rf "$X"; echo target-has-it; exit 0; fi'] + finish + [
                f'rm -rf {q(d)}; mv "$X"/{q(c)} {q(d)}; grep -q {q(exp["digests"][c])} {q(d + "/fingerprint.json")}; '
                f'echo copied $(ls {q(d)}/batch_*.npz | wc -l) batches']
            res = self.copy(helper, self.target, parents, items, {"base": ck_dst}, "; ".join(finish))
            if res.startswith("copied") or res.startswith("target-has-it"):
                p["copied"].append(c)
                self.event(f"{helper} -> {self.target}: {stage} {c}: {res}")
            else:
                self.log(f"{helper} -> {self.target}: {stage} {c}: {res} — retrying next check")

    def assemble(self, stage: str) -> bool:
        a = self.s["assemble"].setdefault(stage, {"jobs": [], "status": "new", "retries": 0})
        t = self.target
        if a["status"] == "done":
            return True
        if a["status"] == "new":
            parts = self.s["parts"][stage]
            missing = [c for h in self.helpers for c in (self.s["assignment"].get(h) or [])
                       if c not in parts.get(h, {}).get("copied", [])]
            cl = self.cluster(t)
            minutes = share_minutes(self.cfg(), self.multi, self.s["scope"], stage, missing, cl.speed) + 30
            n, limit = chain_shape(minutes, cl.max_job_hours)
            if missing:
                self.event(f"{t}: assembling {stage}; computing {len(missing)} undelivered condition(s) itself: {missing}")
                for h in self.helpers:      # free the GPUs of a share given up
                    jobs = parts.get(h, {}).get("jobs") or []
                    if jobs and parts[h].get("status") == "abandoned" and self.cluster(h).alive():
                        self.sh(h, f"scancel {' '.join(jobs)}")
            a["jobs"] = self.submit(t, stage, None, n, limit)
            a["status"] = "running" if a["jobs"] else "new"
            return False
        states = self.job_states(t, a["jobs"])
        if states is None:
            return False
        if any(s == "COMPLETED" and e.startswith("0:") for s, e in states.values()):
            a["status"] = "done"
            self.cancel_waiting(t, states)
            self.event(f"{t}: {stage} assembled")
            return True
        if all(s in ENDED or s.startswith("CANCELLED") for s, _ in states.values()):
            if a["retries"] < int(self.multi.get("max_retries", 3)):
                a["retries"] += 1
                a["status"] = "new"
                self.event(f"{t}: {stage} assemble ended without finishing — resubmitting ({a['retries']})")
            else:
                self.s["phase"] = "error"
                self.event(f"{t}: {stage} assemble keeps failing — stopped; see its job logs")
        return False

    def finish(self) -> bool:
        t = self.target
        f = self.s.setdefault("finish", {"logs": [], "record": False, "packed": False})
        tlog = f'{self.s["env"][t]["logs"]}/{self.run_name}/helpers'
        for h in self.helpers:
            if h in f["logs"]:
                continue
            hl = self.s["env"][h]["logs"]
            res = self.copy(h, t, hl, [("", self.run_name)], {"base": tlog},
                            f'mkdir -p {q(tlog)}/{q(h)}; cp -rp "$X"/{q(self.run_name)}/. {q(tlog)}/{q(h)}/')
            if res.startswith("ok"):
                f["logs"].append(h)
                self.event(f"{h}: job logs copied to {t}")
        if not f["record"]:
            rec = {k: self.s[k] for k in ("run", "scope", "base_config", "assignment", "parts", "assemble", "events", "started")}
            rec["heads"] = {c: e["head"] for c, e in self.s["env"].items()}
            work = self.s["paths"][t]["work_dir"]
            p = self.cluster(t).popen(f"cat > {q(work)}/multi_cluster.json", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            p.communicate(json.dumps(rec, indent=2).encode(), timeout=120)
            f["record"] = p.returncode == 0
        if not f["packed"]:
            r = self.sh(t, f"bash jobs/pack_run.sh {q(self.run_name)} 2>&1 | tail -3", timeout=3600)
            f["packed"] = r.returncode == 0
            self.event(f"{t}: pack_run.sh: {r.stdout.strip()[-200:]}")
        return f["record"] and f["packed"]

    def step(self) -> None:
        ph = self.s["phase"]
        if ph == "preflight" and self.preflight():
            self.s["phase"] = "prepare"
        elif ph == "prepare" and self.prepare():
            self.s["phase"] = "inputs"
        elif ph == "inputs" and self.sync_inputs():
            self.s["phase"] = "calibrate"
        elif ph == "calibrate" and self.stage_parts("calibrate"):
            self.s["phase"] = "calibrate-assemble"
        elif ph == "calibrate-assemble" and self.assemble("calibrate"):
            self.s["phase"] = "calibration-to-helpers"
        elif ph == "calibration-to-helpers" and self.sync_calibration():
            self.s["phase"] = "run"
        elif ph == "run" and self.stage_parts("run"):
            self.s["phase"] = "run-assemble"
        elif ph == "run-assemble" and self.assemble("run"):
            self.s["phase"] = "evaluate"
        elif ph == "evaluate":
            a = self.s["assemble"].setdefault("evaluate", {"jobs": [], "status": "new", "retries": 0})
            if a["status"] == "new":
                a["jobs"] = self.submit(self.target, "evaluate", None, 1, "01:30:00")
                a["status"] = "running" if a["jobs"] else "new"
            else:
                st = self.job_states(self.target, a["jobs"])
                if st is None:
                    pass                        # Slurm or the connection did not answer: next check
                elif any(s == "COMPLETED" for s, _ in st.values()):
                    a["status"] = "done"
                    self.event(f"{self.target}: evaluate done")
                    self.s["phase"] = "finish"
                elif st and all(s in ENDED or s.startswith("CANCELLED") for s, _ in st.values()):
                    a["status"] = "new"
        elif ph == "finish" and self.finish():
            self.s["phase"] = "done"
            self.event(f"run {self.run_name} complete")
        self.save()

    def loop(self) -> None:
        poll = max(60, int(self.multi.get("poll_seconds", 300)))
        (self.dir / "orchestrator.pid").write_text(str(os.getpid()))
        self.log(f"orchestrator started (pid {os.getpid()}), phase {self.s['phase']}, checking every {poll} s")
        try:
            while self.s["phase"] not in ("done", "error"):
                before = self.s["phase"]
                try:
                    self.step()
                except Exception as e:          # keep the run alive across transient faults
                    self.log(f"error in phase {before}: {type(e).__name__}: {e}")
                if self.s["phase"] != before and self.s["phase"] not in ("done", "error"):
                    continue                    # go straight on to the next phase
                if self.s["phase"] not in ("done", "error"):
                    time.sleep(poll)
        finally:
            (self.dir / "orchestrator.pid").unlink(missing_ok=True)
            self.log(f"orchestrator stopped, phase {self.s['phase']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_plan(cfg: dict, multi: dict, scope: str) -> None:
    a = assign(cfg, multi)
    print(f"target {multi['target']}, helpers {', '.join(multi.get('helpers') or [])}")
    for stage in ("calibrate", "run"):
        print(f"\n{stage}: estimated work per cluster (each share runs as chained jobs of max_job_hours)")
        for c, conds in a.items():
            spec = multi["clusters"][c]
            m = share_minutes(cfg, multi, scope, stage, conds, float(spec.get("speed", 1.0)))
            n, lim = chain_shape(m, float(spec.get("max_job_hours", 8)))
            print(f"  {c:10s} {m / 60:5.1f} h   {n} job(s) x {lim}   {', '.join(conds)}")
        total = sum(share_minutes(cfg, multi, scope, stage, conds, 1.0) for conds in a.values())
        longest = max(share_minutes(cfg, multi, scope, stage, conds, float(multi['clusters'][c].get('speed', 1.0)))
                      for c, conds in a.items())
        print(f"  on one cluster: {total / 60:.1f} h; split: about {longest / 60:.1f} h plus the assemble step")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "start", "run"):
        p = sub.add_parser(name)
        p.add_argument("scope", choices=("lean", "full"))
        p.add_argument("--run-name", required=True)
        p.add_argument("--multi", default="config/multi_cluster.yaml")
        p.add_argument("--config", default=None, help="base config (default: the scope's standard one)")
        p.add_argument("--target", default=None, help="the cluster that holds the run (overrides the multi file)")
        p.add_argument("--helpers", default=None,
                       help="comma-separated helper clusters (overrides the multi file; '' for none)")
    p = sub.add_parser("remote-check", help=argparse.SUPPRESS)
    p.add_argument("config")
    p.add_argument("--conditions", default="")
    p.add_argument("--mimic", action="store_true")
    for name in ("status", "stop"):
        p = sub.add_parser(name)
        p.add_argument("run_name")
    p = sub.add_parser("remote-info", help=argparse.SUPPRESS)
    p.add_argument("config")
    p.add_argument("--stage", choices=("calibrate", "run"), default=None)
    p.add_argument("--paths-only", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))

    if args.cmd == "remote-info":
        print(json.dumps(remote_info(args.config, args.stage, args.paths_only)))
        return
    if args.cmd == "remote-check":
        conds = [c for c in args.conditions.split(",") if c]
        print(json.dumps(remote_check(args.config, conds, args.mimic)))
        return
    if args.cmd in ("status", "stop"):
        d = REPO / "multi_runs" / args.run_name
        o = Orchestrator(args.run_name, d)
        if o.s is None:
            sys.exit(f"no multi-cluster run '{args.run_name}' in {d}")
        pid_file = d / "orchestrator.pid"
        pid = int(pid_file.read_text()) if pid_file.exists() else None
        alive = pid is not None and Path(f"/proc/{pid}").exists()
        if args.cmd == "stop":
            if alive:
                os.kill(pid, signal.SIGTERM)
                print(f"stopped orchestrator {pid}; submitted jobs keep running; start again with the same command")
            else:
                print("orchestrator not running")
            return
        print(f"run {o.s['run']} ({o.s['scope']}): phase {o.s['phase']}; orchestrator "
              f"{'running (pid %d)' % pid if alive else 'NOT running'}")
        for stage in ("calibrate", "run"):
            for c, p in (o.s["parts"].get(stage) or {}).items():
                n = len(o.s["assignment"].get(c) or [])
                print(f"  {stage:9s} {c:10s} {p['status']:9s} jobs {','.join(p['jobs']) or '-'}  "
                      f"copied {len(p['copied'])}/{n if c != o.target else '-'}  retries {p['retries']}")
            a = o.s["assemble"].get(stage)
            if a:
                print(f"  {stage:9s} assemble   {a['status']:9s} jobs {','.join(a['jobs']) or '-'}")
        print("  last events:")
        for e in o.s["events"][-6:]:
            print(f"    {e['time']}  {e['event']}")
        return

    from common import load_config
    base = args.config or STANDARD_CONFIG[args.scope]
    multi = yaml.safe_load(open(args.multi))
    if args.target:
        multi["target"] = args.target
    if args.helpers is not None:
        multi["helpers"] = [h for h in args.helpers.split(",") if h]
    if multi["target"] in (multi.get("helpers") or []):
        sys.exit(f"cluster '{multi['target']}' is both the target and a helper")
    for c in [multi["target"]] + list(multi.get("helpers") or []):
        if c not in multi["clusters"]:
            sys.exit(f"cluster '{c}' is not under clusters: in {args.multi}")
    cfg = load_config(base)
    if args.cmd == "plan":
        print_plan(cfg, multi, args.scope)
        return
    d = REPO / "multi_runs" / args.run_name
    o = Orchestrator(args.run_name, d)
    if o.s is None:
        o.init(args.scope, base, args.multi, multi)
        print_plan(cfg, multi, args.scope)
    elif (o.s["scope"], o.s["base_config"]) != (args.scope, base):
        sys.exit(f"run '{args.run_name}' was started as {o.s['scope']} with {o.s['base_config']}")
    elif (args.target and args.target != o.target) or \
            (args.helpers is not None and multi["helpers"] != o.helpers):
        sys.exit(f"run '{args.run_name}' was started with target {o.target} and helpers {o.helpers}; "
                 f"a run keeps them (start a new run name to change them)")
    pid_file = d / "orchestrator.pid"
    if pid_file.exists() and Path(f"/proc/{int(pid_file.read_text())}").exists():
        sys.exit(f"an orchestrator for '{args.run_name}' is already running (pid {pid_file.read_text()})")
    if args.cmd == "start":
        log = open(d / "multi.log", "a")
        p = subprocess.Popen([sys.executable, str(REPO / "src/multi_cluster.py"), "run", args.scope,
                              "--run-name", args.run_name, "--multi", args.multi, "--config", base,
                              "--target", o.target, "--helpers", ",".join(o.helpers)],
                             cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"orchestrator started (pid {p.pid}); log {d / 'multi.log'}; "
              f"python src/multi_cluster.py status {args.run_name}")
        return
    o.loop()


if __name__ == "__main__":
    main()
