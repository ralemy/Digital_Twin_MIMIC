# MIMIC-IV Digital Twin Experiment on Alliance Canada clusters

Runnable implementation of Chapter 5's lean-scope methods — a
data-harmonization agent, a patient-similarity agent, an LLM-based
forecasting agent, a critic/validation agent, three non-LLM baselines
(naive, gradient-boosted trees, LSTM) and the RQ1–RQ3 statistical analysis —
run as Slurm jobs on a Digital Research Alliance of Canada cluster
([Rorqual](https://docs.alliancecan.ca/wiki/Rorqual/en),
[Trillium](https://docs.alliancecan.ca/wiki/Trillium), ...), one H100 80GB
GPU per job. Cluster differences (internet on compute nodes, where jobs may
write, GPU-only partitions) are settings in your profile (section 1), not
code changes.

Everything stays on the cluster: MIMIC-IV is read from disk, and every LLM
call goes to an Ollama server the job starts on its own compute node,
bound to `127.0.0.1`. No cloud service or external API is used.

## Before you start: approvals and permissions

> **This repository assumes no approval or permission has been obtained.**
> Obtain everything below **before running any step on real MIMIC-IV data**.
> Nothing in the code can check these for you. Only the smoke test
> (section 3) can run earlier: it uses synthetic data and a mocked LLM.

1. **Ethics approval.** Ensure appropriate approval for this secondary use
   of MIMIC-IV is obtained from the relevant research ethics board (REB/IRB)
   at your institution — or a documented determination that it is exempt.
   The approval should cover the protocol you actually run, in particular
   the variable panel: the lean (5-variable) and full (19-variable) scopes
   below are different protocols, and approval for one does not
   automatically cover the other. Changes to the panel, cohort or outcomes
   may need an amendment.
2. **Credentialed access to MIMIC-IV.** A PhysioNet account that is
   credentialed, the human-subjects research training PhysioNet requires
   (e.g. CITI "Data or Specimens Only Research"), and the signed Data Use
   Agreement (DUA) for the MIMIC-IV version used (3.1). Every person who
   can read the data or patient-level outputs needs the same.
3. **Storage and handling that meet the DUA** and your institution's data
   security and privacy requirements, plus any conditions of your ethics
   approval:
   - Confirm, with your REB or privacy office if needed, that the Alliance
     cluster and file system you choose are acceptable for MIMIC-IV.
   - Alliance project space (`/project/...`) is readable by your whole
     group. Keep MIMIC-IV and the results where only credentialed people
     can read them (e.g. `chmod -R go-rwx` on those directories), or confirm
     every group member is credentialed.
   - Never send the data, or patient-level outputs, to third-party online
     services. This project's LLMs run locally (Ollama on the compute node,
     `127.0.0.1` only) for this reason; follow PhysioNet's guidance on the
     responsible use of MIMIC data with LLMs and online services.
4. **Alliance access.** An Alliance (CCDB) account, sponsored by your
   supervisor or PI; membership in an allocation (`def-<pi>`, used as
   `slurm.account` in your profile, section 1); and access to the cluster itself — some
   clusters (e.g. Rorqual) require requesting access in CCDB and accepting
   the site's agreements first. Follow the Alliance's usage policies.
5. **Model licences.** Read and accept the terms of each model before
   downloading it: Gemma Terms of Use (Gemma 3), Health AI Developer
   Foundations terms (MedGemma), Meta Llama 3 Community Licence and
   Acceptable Use Policy (Llama 3, Med42), Apache 2.0 (Qwen2.5,
   Baichuan-M2). Confirm their terms permit your use.
6. **Weights & Biases (optional, off by default).** If you turn it on
   (`environment.disable_wandb: false` in your profile), the jobs send
   aggregate metrics (no patient-level values, see section 4) to W&B, an
   external service. Do so only if your ethics approval and data governance
   permit it, and accept W&B's terms.

This guide takes you from a fresh clone to results:

1. [Set up: profile, `setup_bash.sh`, `source ~/.bashrc`](#1-set-up-profile-setup_bashsh-source-bashrc)
2. [What the setup does — and how to redo it](#2-what-the-setup-does--and-how-to-redo-it)
3. [Run the smoke test](#3-run-the-smoke-test)
4. [Get the data and the models](#4-get-the-data-and-the-models)
5. [Run the experiment step by step](#5-run-the-experiment-step-by-step)
6. [Or run it all with `jobs/run_all.sh`](#6-or-run-it-all-with-jobsrun_allsh)

followed by [monitoring](#7-monitoring-jobs), [where the results
are](#8-where-the-results-are) and [reference material](#reference).

## Two scopes — pick deliberately

| Config | Panel |
|---|---|
| `config/config_alliance_lean.yaml` ("lean", the default) | 5 variables: heart rate, respiratory rate, SpO2, MAP, lactate |
| `config/config_alliance_full.yaml` ("full") | the same 5 plus 3 vitals and 11 labs (19 variables) |

The two configs are different scopes, not hardware settings: they run the
same code with a different variable panel and write to separate results
directories. **Run only the scope(s) your ethics approval covers** (see
"Before you start", item 1). The full panel is a proposed expansion —
review its `variables:` before treating it as final.

(`config/config_local.yaml` is the original small-scale configuration for a
single 11GB GTX 1080 Ti workstation; it is not used on the clusters.)

---

## 1. Set up: profile, `setup_bash.sh`, `source ~/.bashrc`

**Your profile, `~/.config/dt_profile.yml`, is the only file you edit.** It
holds every location and setting that differs between clusters or users —
where MIMIC-IV, the results and the Ollama models go, your Slurm account —
and your PhysioNet login. Everything else is done by `jobs/setup_bash.sh`.
Alliance clusters don't share home directories, so do this once on each
cluster you use, on a login node:

```bash
git clone <repository URL> <where you want it>     # this becomes $DT_REPO
cd <where you want it>

# 1. Your profile: copy the template, make it private, fill it in (below)
cp config/profile.sample.yml ~/.config/dt_profile.yml
chmod 600 ~/.config/dt_profile.yml                  # required: it holds your PhysioNet password
nano ~/.config/dt_profile.yml                       # or any editor

# 2. The setup: symlinks, Ollama, .venv, ~/.bashrc (section 2)
bash jobs/setup_bash.sh

# 3. Load the variables it wrote into ~/.bashrc
source ~/.bashrc
echo "$DT_REPO  $DT_MIMIC_DIR  $DT_RESULTS_DIR  $DT_OLLAMA_MODELS  $SBATCH_ACCOUNT"
```

That's all the smoke test (section 3) and the other jobs need. New login
shells get the variables from `~/.bashrc` automatically.

### What to fill in

| Profile key | What it is | Sets | Repo symlink |
|---|---|---|---|
| `slurm.account` | your allocation, e.g. `def-<pi>`; `sshare -U -u $USER` lists yours (use the `def-...` one — GPU jobs are charged to its `_gpu` counterpart automatically) | `DT_ACCOUNT`, exported as `SBATCH_ACCOUNT`/`SALLOC_ACCOUNT` | |
| `slurm.gpu_jobs_only` | `true` where every job must take a GPU and may not ask for memory (Trillium's GPU subcluster); `jobs/submit.sh` then gives every job one GPU and drops `--mem`. Empty: `true` on Trillium, `false` elsewhere | `DT_GPU_JOBS_ONLY` | |
| `paths.mimic_dir` | where the MIMIC-IV files (`hosp/`, `icu/`) are downloaded | `DT_MIMIC_DIR` | `mimic-iv` |
| `paths.results_dir` | where results are recorded: `<results_dir>/mimic-iv-twin-work` (lean) and `<results_dir>/mimic-iv-twin-work-full` (full). Must be on `$SCRATCH` when `workers_can_write_repo` is `false` | `DT_RESULTS_DIR` | |
| `paths.logs_dir` | where job `.out` files and Ollama logs go (empty: `$DT_REPO/logs`). Must be on `$SCRATCH` when `workers_can_write_repo` is `false` | `DT_LOGS_ROOT` | |
| `paths.ollama_models` | where the Ollama models are downloaded | `DT_OLLAMA_MODELS`, `OLLAMA_MODELS` | `ollama-models` |
| `paths.ollama_bin` | where Ollama is installed; must end in `/bin` (default `~/ollama-local/bin`, ~2 GB) | `DT_OLLAMA_BIN`, first on `PATH` | |
| `environment.ollama_version` | the Ollama release to install (empty: 0.34.4, the version this project was run with) | `DT_OLLAMA_VERSION` | |
| `environment.modules` | modules every job loads before `.venv` (empty: `StdEnv/2023 python/3.11`) | `DT_MODULES` | |
| `environment.disable_wandb` | `true` (default) keeps Weights & Biases off whatever the configs say; `false` allows it (section 4) | `DT_WANDB_DISABLED`, `WANDB_MODE=disabled` | |
| `environment.workers_have_internet` | whether the cluster's compute nodes reach the internet, e.g. `false` on Rorqual and Trillium. With `false`, downloads run on a login node (section 4). Empty: `false` on Trillium, `true` elsewhere | `DT_WORKER_INTERNET` | |
| `environment.workers_can_write_repo` | whether compute nodes can write the repo, `$HOME` and `/project`. With `false`, the setup refuses a `results_dir` or `logs_dir` there, and makes the tuned configs (`config/<config>_tuned.yaml`) symlinks into `$DT_RESULTS_DIR/tuned-configs`. Empty: `false` on Trillium, `true` elsewhere | `DT_WORKER_WRITES_REPO` | `config/*_tuned.yaml` (when `false`) |
| `physionet.username`, `physionet.password` | your PhysioNet account, credentialed for MIMIC-IV — read only by the download job | (not exported) | |

Each `true`/`false` key also accepts `yes`/`no`, `on`/`off` and `1`/`0`;
anything else stops the setup with a message. The cluster is taken from
`$CC_CLUSTER`, or from the login node's name on Trillium; elsewhere it is
`local` and every empty flag gets its non-Trillium default.

Paths must be absolute; `~`, `$HOME`, `$SCRATCH`, `$USER` and `$DT_REPO`
(where the repo is cloned) are expanded. The template has an example for
each. Guidelines from the Alliance storage policies:

- **MIMIC-IV** (~10 GB for `hosp/` + `icu/`) and **results** belong on
  project space (`~/projects/<account>/...`): backed up, and not purged.
  Avoid `$HOME` itself for MIMIC-IV (small quota).
- **Exception — Trillium:** compute nodes can only read `$HOME` and
  `/project`, so `results_dir` and `logs_dir` go on `$SCRATCH` (e.g.
  `$SCRATCH/dt-results`, `$SCRATCH/dt-logs`). Scratch is not backed up and
  is purged: copy finished results to project space yourself.
- **Ollama models** (~190 GB for all six) fit best on scratch (`$SCRATCH`,
  large quota, not backed up). Scratch is purged after a period of
  inactivity; if models go missing, download them again (section 4).
- Check free space first with `diskusage_report`.
- **Data custody:** the locations you choose for MIMIC-IV and results must
  meet the DUA and your approval's conditions, and be readable only by
  credentialed people ("Before you start", item 3).

Keep the profile private: it lives in your home directory, never in the
repo or in `/project` (shared with your group). Everything that reads it
refuses a profile that others can read (`chmod 600` fixes it), and the
password is never exported to the jobs' child processes.

---

## 2. What the setup does — and how to redo it

`bash jobs/setup_bash.sh` reads your profile — and stops, saying exactly
which key to fix, if it is missing, readable by others or incomplete — then:

1. **Symlinks.** Creates `mimic-iv` → `$DT_MIMIC_DIR` and `ollama-models` →
   `$DT_OLLAMA_MODELS` in the repo (or re-points them if they lead
   elsewhere), plus their target directories and `$DT_RESULTS_DIR`. The configs reach MIMIC-IV through `mimic-iv`.
   With `workers_can_write_repo: false`, each `config/<config>_tuned.yaml`
   also becomes a symlink into `$DT_RESULTS_DIR/tuned-configs`, so step 3b
   can write it from a compute node.
2. **Ollama, without root.** Downloads the Ollama release
   `environment.ollama_version` (~1.4 GB, from GitHub; login nodes have
   internet access) and unpacks it into the parent of `paths.ollama_bin`
   (`bin/` and `lib/`). Skipped if an `ollama` binary is already there. The
   GPU jobs start their own `ollama serve` on the compute node, on a port
   derived from the job id (`jobs/ollama_lib.sh`) — don't run it on the
   login node.
3. **`.venv`**, the Alliance way (below). A valid `.venv` — its python
   runs, matches the loaded `python/` module and has every pinned package of
   `requirements.txt` — is kept; anything else (e.g. half-built by a failed
   run) is removed and rebuilt.
4. **`~/.bashrc`.** Writes the final values of every variable
   (`DT_REPO`, `DT_MIMIC_DIR`, `DT_RESULTS_DIR`, `DT_OLLAMA_MODELS`,
   `OLLAMA_MODELS`, `SBATCH_ACCOUNT`, ..., and `PATH` with the ollama
   binary) into a marked block at its end, between
   `# >>> mimic-iv digital twin ...` and `# <<< mimic-iv digital twin <<<`.
   A previous block is replaced, never duplicated; if nothing changed the
   file isn't touched, otherwise the old file is kept as
   `~/.bashrc.dt-backup`. Lines elsewhere in `~/.bashrc` that set the same
   variables are listed so you can delete them (the block comes last and
   wins). The job scripts carry no `--account` of their own, so `sbatch`
   relies on this `SBATCH_ACCOUNT`.

**Log and lock.** Everything the setup prints is also written to
`setup.lock` in the repo base (git-ignored). While that file exists, the
setup refuses to run again, so it can't be repeated by accident. To redo it
— for example after changing your profile:

```bash
rm setup.lock && bash jobs/setup_bash.sh && source ~/.bashrc
```

A failed run leaves no lock: its log is kept as `setup-failed-<time>.log`
(also git-ignored), and you can run the setup again once the cause is fixed.
Running it again is safe: it only changes what differs from your profile.
An existing Ollama install is left as it is (delete `bin/ollama` to
reinstall, e.g. for another `ollama_version`); `.venv` is rebuilt only if
it isn't valid (delete it to force a rebuild).

**Trillium.** The setup recognises it and applies its rules (each is a
profile key you can also set yourself): compute nodes have no internet
(`workers_have_internet`), can't write `$HOME` or `/project`
(`workers_can_write_repo` — `paths.results_dir` and `paths.logs_dir` must
be on `$SCRATCH`, and the tuned configs become symlinks into
`results_dir`), and GPU-subcluster jobs must take a GPU and may not ask for
memory (`slurm.gpu_jobs_only`). Run `jobs/run_all.sh` from the GPU login
node (`trig-login01`); every job then gets one H100 (a quarter node, 24
cores, ~188 GiB) and no `--mem`, including the short CPU-only steps. Submit
single jobs with `bash jobs/submit.sh jobs/<job>.sh ...` rather than
`sbatch`: it applies these rules, and on every cluster sends the `.out`
file to `paths.logs_dir`.

Jobs never need the lock or `~/.bashrc`: each one re-reads the profile
itself (through `jobs/load_profile.sh`, which sources `jobs/setup_bash.sh`
in read-only mode) and prints the cluster, account and resolved locations
at the top of its `.out` file. To use another profile, pass
`--profile <file>` among a job's arguments.
[docs/load_profile.md](docs/load_profile.md) explains the mechanics.

### Python packages and modules

The setup builds `.venv` with:

```bash
module load $DT_MODULES                  # environment.modules; default StdEnv/2023 python/3.11
virtualenv --no-download .venv
.venv/bin/pip install --no-index --upgrade pip
.venv/bin/pip install --no-index -r requirements.txt
```

`--no-index` installs from the Alliance wheelhouse, not PyPI.
`requirements.txt` lists only the top-level packages (duckdb, fastparquet,
scikit-learn, torch, wandb, pytest); their dependencies (pandas, numpy,
scipy, ...) come with them. Its header also lists the modules that are
loaded instead of pip-installed: `python/3.11`, and optionally
`gcc arrow/25.0.0`, which gives pandas pyarrow (add it to
`environment.modules` in your profile). Don't switch arrow on in the
middle of an experiment: string columns then read as pandas `str` instead
of `object`.

---

## 3. Run the smoke test

Before the smoke test you need only section 1's three steps. The smoke
test needs no
GPU, no Ollama and no MIMIC-IV data: it fabricates a small synthetic cohort,
mocks the LLM, and runs every condition end to end.

```bash
cd "$DT_REPO"
bash jobs/submit.sh jobs/smoke_test.sh   # 4 CPUs, 8 GB, 15 min limit (on Trillium: one GPU, no --mem)
sq                                            # your jobs
```

When it ends, `$DT_LOGS_ROOT/mimic-twin-smoke-<jobid>.out` (`paths.logs_dir`;
`logs/` in the repo by default) should finish with:

```
[smoke_test] INFO: SMOKE TEST PASSED: full pipeline ran end to end with no errors.
```

If it fails, the `.out` file says why — usually a missing package (rebuild
`.venv`) or a setting in your profile. Fix it now;
it's much cheaper than finding out after a multi-hour job.

---

## 4. Get the data and the models

### MIMIC-IV

You need credentialed MIMIC-IV access and an approved, compliant place to
store the data ("Before you start", items 1–3), and `physionet.username` /
`physionet.password` filled in in your profile (section 1). Then download
into `$DT_MIMIC_DIR`:

```bash
bash jobs/submit.sh jobs/prep1_download_mimic.sh   # MIMIC-IV 3.1; resumable, re-submit after a timeout
```

It checks every `hosp/` and `icu/` file against PhysioNet's
`SHA256SUMS.txt` before and after downloading, deletes any that fail, and
downloads them again, so running it again also verifies and repairs an
existing copy (e.g. if step 2 fails with DuckDB's "Input is not a GZIP
stream"). If your profile says `workers_have_internet: false`, run it on the
login node instead (see "Clusters without internet on compute nodes" below).

The result is the standard PhysioNet layout, reached by the configs
through the repo's `mimic-iv` symlink:

```
$DT_MIMIC_DIR/
  hosp/   patients.csv.gz  admissions.csv.gz  labevents.csv.gz  d_labitems.csv.gz  ...
  icu/    icustays.csv.gz  chartevents.csv.gz  d_items.csv.gz  inputevents.csv.gz  ...
```

The password never leaves the profile except into a temporary mode-600
`wgetrc` inside the job, deleted when it ends.

### The Ollama models

The configs compare six models (~190 GB in total):

| Model | Alias | Size | Source |
|---|---|---|---|
| `qwen2.5:32b-instruct-q8_0` (variant `qwen2_5_32b`) | `qwen2.5:32b` | ~35 GB | Ollama library |
| Baichuan-M2-32B Q8_0 | `baichuan-m2:32b` | ~35 GB | Hugging Face (bartowski) |
| `gemma3:27b` | | ~17 GB | Ollama library |
| MedGemma 27B text Q4_K_M | `medgemma:27b` | ~17 GB | Hugging Face (unsloth) |
| `llama3:70b-instruct-q4_K_M` | | ~43 GB | Ollama library |
| Llama3-Med42-70B Q4_K_M (primary, `llm.model`) | `med42:70b` | ~43 GB | Hugging Face (mradermacher) |

```bash
bash jobs/submit.sh jobs/prep2_download_models.sh config/config_alliance_lean.yaml
```

(With `workers_have_internet: false`, run it on the login node — below.)

It pulls only what's missing from `$DT_OLLAMA_MODELS` and creates the
aliases; re-submit after a timeout and it resumes. None of the downloads
needs a login, but accept each model's licence first ("Before you start",
item 5). The GPU jobs
never download: they stop with the commands to run if a model is missing.

### Clusters without internet on compute nodes (e.g. Rorqual, Trillium)

`prep1` and `prep2` download from the internet. Where compute nodes have
access, they run as Slurm jobs. On clusters whose compute nodes have none
(Rorqual, Trillium), set `environment.workers_have_internet: false` in your
profile (the default on Trillium) and redo the setup. Then:

- **Run the downloads on a login node, with `bash` instead of `sbatch`.** They
  take hours, so start them inside `tmux` (or `screen`) so a dropped
  connection doesn't stop them; both resume where they left off if
  interrupted:

  ```bash
  tmux new -s download
  cd "$DT_REPO"
  bash jobs/prep1_download_mimic.sh                        # MIMIC-IV, ~10 GB
  bash jobs/prep2_download_models.sh config/config_alliance_lean.yaml   # models, ~190 GB
  # Ctrl+B, D detaches; tmux attach -t download returns
  ```

  Their output goes to the terminal; `| tee "$DT_LOGS_ROOT/prep1-login.out"` keeps a copy.
- **Submitted as a Slurm job by mistake,** they stop at once with that advice.
- **`jobs/run_all.sh`** runs its `models` stage on the login node by itself,
  logging to the run's folder in `$DT_LOGS_ROOT`; `--plan` shows it.

Login nodes are shared: these downloads are network-bound and light on CPU
and memory, and on Rorqual the login node is also the documented data-transfer
node. Alternatively, copy the files from a cluster that already has them with
[Globus](https://docs.alliancecan.ca/wiki/Globus) (e.g. from another
cluster's collection to `alliancecan#rorqual`) into `$DT_MIMIC_DIR` and `$DT_OLLAMA_MODELS`; `prep2`
then finds nothing to download.

### Live metrics on Weights & Biases (optional)

W&B is **off by default**: your profile's `environment.disable_wandb: true`
keeps it off whatever the configs say. It sends aggregate metrics to an
external service, so turn it on only if your ethics approval and data
governance allow it ("Before you start", item 6): set
`environment.disable_wandb: false` in your profile (and redo the setup), and
keep `logging.wandb.enabled: true` in the config. The tune, calibrate, run
and evaluate jobs then log aggregate metrics to the W&B project
`mimic-iv-digital-twin`. Set it up once on a login node:

```bash
source .venv/bin/activate && wandb login && chmod 600 ~/.netrc
```

- **Logged:** per-batch progress, fallback and error counts by type,
  per-condition sMAPE/MAE/plausibility/coverage/fill rate, tuning scores,
  calibration coverage, the analysis' means/CIs/p-values, and the job's
  CPU/memory/GPU use. Runs are named `<config>-<stage>-<jobid>`.
- **Never sent** (`src/tracking.py`): log lines, model output, per-patient
  values, stay ids or files. A failed forecast is reported only by type,
  e.g. "malformed forecast - see logs for details".
- If W&B isn't reachable (not logged in, or no internet on the compute
  nodes, as on Rorqual and Trillium), the job logs one warning and runs without it; on
  such clusters keep `disable_wandb: true`.

---

## 5. Run the experiment step by step

Every job is submitted from the repository base with `bash jobs/submit.sh`
and takes the config as its first argument; without one, every job (and
every `src/` script's `--config-file`) uses `config/config_alliance_lean.yaml`.
`jobs/submit.sh` passes `sbatch` options given before the script
(`--option=value` form), sends the `.out` file to `paths.logs_dir`, and
applies `slurm.gpu_jobs_only`. Plain `sbatch jobs/<job>.sh` still works
where compute nodes can write the repo (the `.out` file then goes to
`logs/` in the repo), but not on Trillium:

```bash
cd "$DT_REPO"
CONFIG=config/config_alliance_lean.yaml             # lean scope
# CONFIG=config/config_alliance_full.yaml  # full scope — only if your approval covers it
TUNED=${CONFIG%.yaml}_tuned.yaml                 # written by step 3b
```

| Step | Job | Resources (time limit) | Estimated work, lean / full |
|---|---|---|---|
| 1 Resolve variables to itemids | `step1_resolve_items.sh $CONFIG` | 8 CPU, 32 GB (1h15) | 5 min / 5 min |
| 2 Extract cohort and panel | `step2_extract_cohort.sh $CONFIG` | 12 CPU, 192 GB (2h) | 10 min / 20 min |
| 3b Tune hyperparameters | `step3b_tune.sh $CONFIG` | 1 H100, 12 CPU, 64 GB (8h) | 3.5 h / 12 h |
| 3c Calibrate intervals | `step3c_calibrate.sh $TUNED` | 1 H100, 12 CPU, 64 GB (8h) | 10 h / 36 h |
| 3 Run every condition | `step3_run_experiment.sh $TUNED` | 1 H100, 12 CPU, 64 GB (8h) | 14 h / 50 h |
| 4 Statistical analysis | `step4_evaluate_results.sh $TUNED` | 4 CPU, 16 GB (1h30) | 30 min / 30 min |

The resources are the jobs' own `#SBATCH` lines, a conservative one-GPU
share of a node. With
`slurm.gpu_jobs_only` (Trillium) every job, the CPU-only ones included,
runs on a quarter node instead: one H100, 24 cores, ~188 GiB, no `--mem`.
Estimates are from [docs/runtime_estimates.md](docs/runtime_estimates.md)
(at `OLLAMA_NUM_PARALLEL=8` with flash attention). Each step needs the
previous one's output on disk, so start the next only after the previous
one has **finished successfully** (check its `.out` file).

### Step 1 — resolve the variable panel

```bash
bash jobs/submit.sh jobs/step1_resolve_items.sh $CONFIG
```

Writes `<work dir>/cache/item_mapping.json`, where the work dir is
`$DT_RESULTS_DIR/mimic-iv-twin-work` (lean) or `...-full` (full). **Open
and read it before continuing**: it lists every `d_items`/`d_labitems` row
matched for each panel variable. Itemids are resolved by label pattern
against your files rather than hardcoded (they change across MIMIC-IV
releases, and e.g. arterial vs. non-invasive MAP differ), but check it once
and edit the JSON to drop or add an itemid.

### Step 2 — extract the cohort and panel

```bash
bash jobs/submit.sh jobs/step2_extract_cohort.sh $CONFIG
```

One filtered pass each over `chartevents` and `labevents` (the two largest
tables) with DuckDB, cached as Parquet. Writes `<work dir>/cohort.parquet`
and `panel_long.parquet`. CPU/RAM-bound; DuckDB spills to the node-local
`$SLURM_TMPDIR` past its memory limit. On clusters with small node-local
disks, add `--tmp=<N>G` if it fails with "No space left on device".

### Step 3b — tune

```bash
bash jobs/submit.sh jobs/step3b_tune.sh $CONFIG
```

A sequential search (grid in `config/tuning_grid.yaml`) on a 128-patient
*tuning* subset of the validation split — the test split is never read. It
tunes the GBM and LSTM baselines (so the comparison stays fair), the
similarity context and k, the forecasting prompt, temperature and the
critic, keeping a change only if it lowers mean per-patient sMAPE by at
least `min_improvement` without exceeding `max_fallback_rate`. Writes
`$TUNED` (e.g. `config/config_alliance_lean_tuned.yaml`) and
`<results dir>/tuning/` (`trials.csv`, `rounds.json`). The tuned config
writes to `results_tuned` and `checkpoints_tuned`, so untuned results are
kept. With `workers_can_write_repo: false`, `$TUNED` is a symlink and the
file itself is in `$DT_RESULTS_DIR/tuned-configs/`.

`$TUNED` is git-ignored (`config/*_tuned.yaml`), like any config a script
writes: it holds the tuning results of the cluster that ran step 3b, and
checkpoints made with one cluster's tuned settings refuse another's
(`CheckpointMismatch`). Never commit it; to change a setting in it, edit the
cluster's own copy or re-run tuning.

### Step 3c — calibrate the prediction intervals

```bash
bash jobs/submit.sh jobs/step3c_calibrate.sh $TUNED
```

Runs every condition on the remaining 322 validation patients (the
*calibration* subset) and fits one split-conformal scale factor per
condition and variable. Writes `<results dir>/calibration.json`; step 4
applies it to the test forecasts.

### Step 3 — run the experiment

```bash
bash jobs/submit.sh jobs/step3_run_experiment.sh $TUNED
```

Fits the baselines, builds the similarity index, and runs every condition
in the config over the held-out test split, calling the local LLMs. Writes
raw forecast arrays (`*_raw.npz`) and per-condition summaries
(`all_conditions_summary.csv`, rewritten after each finished condition) to
the tuned results dir.

### Step 4 — statistical analysis

```bash
bash jobs/submit.sh jobs/step4_evaluate_results.sh $TUNED
```

RQ1 (orchestration vs. single model), RQ2 (critic ablation: plausibility
violation reduction plus the accuracy trade-off) and RQ3 (stable vs.
deteriorating subgroup), Chapter 5 Section 5.6, with paired tests and
bootstrap 95% CIs resampling patients, plus model-variant and extra
comparisons and interval coverage before/after calibration. Writes
`statistical_analysis.json` to the tuned results dir.

### Jobs longer than 8 hours: resubmit or chain

Steps 3b, 3c and 3 checkpoint as they go (per batch of 32 patients; GBM per
variable; LSTM every few epochs; tuning caches every scored setting). When a
job hits its time limit, **submit the same command again** and it continues
where it left off, losing at most one batch. To queue continuations up
front, chain them with `afterany` — each starts when the previous ends,
however it ended, and exits quickly if nothing is left to do:

```bash
J=$(bash jobs/submit.sh --parsable jobs/step3_run_experiment.sh $TUNED)
J=$(bash jobs/submit.sh --parsable --dependency=afterany:$J jobs/step3_run_experiment.sh $TUNED)
```

Lean needs about 2 jobs each for steps 3c and 3; full needs about 2 for 3b,
6 for 3c and 8 for 3. Arguments after the config go to the Python script:
`--full-refresh` discards the checkpoints and starts over. A checkpoint
records the settings its predictions depend on: variables, cohort, the
model's `llm` settings, the `similarity_agent`, `critic_agent` and
`forecasting_agent` sections, and markers for rule changes in the code
(e.g. the JSON-schema output format). Results made under different
settings are not reused — the tuning cache simply recomputes them. Other
code changes, including edits to the prompt text in `src/`, are **not**
detected — use `--full-refresh` after those.

### The full variable set

The steps are identical with `CONFIG=config/config_alliance_full.yaml`,
which writes to `$DT_RESULTS_DIR/mimic-iv-twin-work-full` so it never
overwrites the lean results. Run it only if your ethics approval covers
the 19-variable panel. Expect a
smaller eligible cohort (19 variables must reach 50% coverage) and about
3.7× the LLM time.

---

## 6. Or run it all with `jobs/run_all.sh`

`jobs/run_all.sh` runs every stage for one scope as Slurm jobs —
models → resolve → extract → tune → calibrate → run → evaluate, with
calibrate/run/evaluate on the tuned config — and resubmits from
checkpoints when a job runs out of time. Run it on a **login node**, not
with `sbatch`:

```bash
cd "$DT_REPO"
bash jobs/run_all.sh lean --plan                 # show the jobs and time limits it would submit
bash jobs/run_all.sh lean                        # start a new run with a random name, e.g. sassy_hammer (the scope defaults to lean)
bash jobs/run_all.sh lean --run-name sassy_hammer   # resume / re-attach to that run (or start one with your own name)
bash jobs/run_all.sh lean --unattended           # a whole run with no one watching
bash jobs/run_all.sh full                        # full scope — only if your approval covers it
bash jobs/run_all.sh lean --status               # stage status, job ids, whether the driver is alive (latest run)
bash jobs/run_all.sh lean --stop                 # stop the driver; submitted jobs keep running (latest run)
```

**Run names.** Every run has a name: `--run-name <name>`, or a new random
one (`funny_rabbit`, `sassy_hammer`, ...) that the driver prints when it
starts. Everything of a run lives under its name, so runs never mix and
calls with the same name resume the same run:

| What | Where |
|---|---|
| the run's config (a copy of the base config, rewritten when a driver starts) and its tuned config | `tuned_configs/<run>/` |
| work, cache, results, checkpoints | `$DT_RESULTS_DIR/mimic-iv-twin-work/<run>/` (full: `mimic-iv-twin-work-full/<run>/`) |
| driver log, job `.out` files, Ollama logs | `$DT_LOGS_ROOT/<run>/` |
| driver state | `run_all/<run>/` |

A new name starts from scratch, including resolve (and its review) and
extract. `--status`, `--stop` and `--plan` without `--run-name` use the
scope's latest run. Drivers of different runs can run side by side.

Other options: `-i <seconds>` (how often Slurm is checked, default 120, at
least 60), `--redo-extract` (re-run resolve and extract even if their
outputs exist), `--redo calibrate,run,evaluate` (re-run finished stages,
e.g. after adding conditions — checkpoints mean only new work is computed),
`--profile <file>` (passed on to every job). The driver submits through
`jobs/submit.sh`, so your profile's cluster settings apply; on Trillium,
start it on the GPU login node (`trig-login01`).

**How it behaves.**
- **Downloads:** the `models` stage runs prep2 as a Slurm job, or on the login
  node itself when your profile says `workers_have_internet: false`.
- **Resolve review:** resolve and extract are skipped if their outputs
  exist. After resolve runs, the driver stops so you can review
  `item_mapping.json` (step 1 above); run the same command again to
  continue.
- **Job sizing:** each stage's estimated work is split into jobs of at most
  7 h of work, each with a time limit 1 h longer (8 h at most). A stage's
  jobs are submitted together as a chain: job *k+1* depends on `afternotok`
  of the earlier ones, so it runs only if they ran out of time or failed.
  When a job completes, the driver cancels the rest of the chain (and,
  except on Trillium, whose `sbatch` rejects `--kill-on-invalid-dep`, so
  does Slurm — a backstop if the driver has died; on Trillium, cancel such
  leftover pending jobs yourself). If a chain
  runs out without finishing, one more job is submitted, up to 3 times. Two
  failed jobs in a row stop the driver.
- **Unattended (`--unattended`):** a stage that stops on failures is
  resubmitted from its checkpoints after a cooldown (15 min, doubling up to
  2 h, at most 5 times); time-outs get up to 6 extra jobs; and it doesn't
  pause for the resolve review (review it afterwards). Only a job cancelled
  by you or an administrator, or the retries running out, stops it. A whole
  lean run is about 46 hours of work plus queue waits.
- **Detached and resumable:** the driver detaches from your terminal
  (`setsid nohup`) and logs to `$DT_LOGS_ROOT/<run>/run_all-<scope>.log`,
  which the command then follows; Ctrl+C or a dropped SSH connection stops
  only the following. Running it again with the same `--run-name`
  re-attaches, or, if the driver died (login node rebooted, `--stop`),
  starts a new one that picks up the submitted jobs from
  `run_all/<run>/<scope>.state`. Jobs keep running under Slurm either
  way.
- **Cluster etiquette:** one driver per scope and run name; Slurm is queried once per
  interval through `jobs/monitor-job.sh`; it refuses to run inside a job.
  Inside `tmux`, `--foreground` works too.

---

## 7. Monitoring jobs

```bash
sq                                   # your queued and running jobs
bash jobs/monitor-job.sh <jobid>     # follow one job: state, why it's pending, its latest output line
seff <jobid>                         # after it ends: CPU/memory actually used
```

Job output goes to `paths.logs_dir` (`$DT_LOGS_ROOT`; default `logs/` in
the repo, git-ignored; on `$SCRATCH` on Trillium), never the repo base:

- **Jobs you submit with `jobs/submit.sh`** write
  `$DT_LOGS_ROOT/<job-name>-<jobid>.out`, and the GPU jobs also
  `ollama-<jobid>.log` next to it.
- **Jobs submitted by `run_all.sh`** write both into one folder per run
  name, `$DT_LOGS_ROOT/<run>/`, next to the driver's own log, so a run's
  logs stay together. `--status` shows it. The driver's state stays in
  `run_all/<run>/`.

The first lines of every `.out` show the
cluster, account and resolved locations.

The download jobs (prep1, prep2) report their progress in their log as plain
lines — one per 10 % of each file or model, or at least once a minute — so
`tail -f $DT_LOGS_ROOT/<job>.out` or `monitor-job.sh` shows how far they are. Run by
hand in a terminal on a login node, they show the usual live progress bar
instead (`jobs/progress_lib.sh`; `DT_PROGRESS_STEP` and `DT_PROGRESS_SECONDS`
change the interval).

## 8. Where the results are

For the lean scope (full: replace `mimic-iv-twin-work` with
`mimic-iv-twin-work-full`); a `run_all.sh` run has all of this one level
down, in `mimic-iv-twin-work/<run>/`, and its configs in `tuned_configs/<run>/`:

```
$DT_RESULTS_DIR/mimic-iv-twin-work/
  cache/item_mapping.json            step 1 — review it
  cohort.parquet, panel_long.parquet step 2
  results/tuning/                    step 3b (trials.csv, rounds.json)
  results_tuned/calibration.json     step 3c
  results_tuned/all_conditions_summary.csv, *_raw.npz      step 3
  results_tuned/statistical_analysis.json                   step 4 — RQ1–RQ3
  checkpoints/, checkpoints_tuned/   resume state (safe to delete once finished)
$DT_RESULTS_DIR/tuned-configs/       the tuned configs, when workers_can_write_repo is false
                                     (tuned_configs/ in the repo is then a link to it)
```

On Trillium `$DT_RESULTS_DIR` is on `$SCRATCH`, which is purged: copy the
finished `mimic-iv-twin-work` folder to project space (Globus or `rsync`).

---

## Reference

### Models and agent combinations

The configs compare six models in three general/medical pairs: Qwen2.5-32B / Baichuan-M2-32B,
Gemma 3 27B / MedGemma 27B, and Llama 3 70B / Med42-70B. Each runs
`single_model_llm` and `full_pipeline`; the primary model also runs the RQ
ablations (`full_pipeline_no_critic`, `full_pipeline_no_similarity`). The
primary model is Med42-70B (`llm.model`) since the second Trillium run, in
which its pipeline was the most accurate LLM condition and Qwen2.5-32B, the
earlier primary, was the one model the pipeline didn't help. Qwen is still
compared, as the `qwen2_5_32b` variant.

Two more kinds of condition make the comparison honest (see
`docs/evaluation.md`, section 5): stronger persistence baselines
(`recent_mean`, `persistence_blend`), and `ensembles` such as
`full_pipeline+lstm`, the mean of two conditions' forecasts, built from their
saved results at no GPU cost.

Any LLM condition can run with another model by appending `@<variant>`,
e.g. `full_pipeline@medgemma`, where the variant is defined under
`llm.variants` (it inherits every `llm` key and overrides what it lists).
Two more conditions test whether the agents should use different models:

- `full_pipeline_clip_critic@medgemma`: MedGemma forecasts, and
  out-of-range values are simply clipped (no LLM critic).
- `full_pipeline@medgemma_gemma3_critic`: MedGemma forecasts, Gemma 3 is the
  critic (`critic_variant: <variant>` under `llm.variants`).

`evaluation.extra_comparisons` lists the condition pairs compared for this;
they appear under `extra_comparisons` in `statistical_analysis.json`. Before
comparing models' accuracy, check the two failure columns in
`all_conditions_summary.csv`: `llm_fallback_rate` (whole forecast replaced
by the naive one) and `llm_filled_rate` (per variable, see "Missing
variables" below).

### Answering the RQs with another model

RQ1–RQ3 are answered for each model in `evaluation.rq_models`; the default
`[primary]` is the pre-specified analysis. To also answer them with the
best-performing model, choose it on validation data, not the test set:

```bash
source .venv/bin/activate
python src/select_rq_model.py --config-file config/config_alliance_lean_tuned.yaml --write-config
bash jobs/run_all.sh lean --unattended --redo calibrate,run,evaluate
```

`select_rq_model.py` ranks `full_pipeline` for each model by per-patient
sMAPE on the calibration patients, excludes models whose fallback rate is
above the tuning cap, and picks the best one only if it is significantly
better than the primary (95% paired-bootstrap CI below 0). The rule and
ranking go to `rq_model_selection.json`; `--write-config` adds the chosen
model and the conditions its RQ2 needs. The chosen model keeps the settings
tuned for the primary — state that as a limitation, or re-tune.

### How the Alliance configs are sized

Compared with the 1080 Ti `config_local.yaml` (same panel, cohort criteria and
conditions): a larger cohort (`max_patients` 3000 vs 300; `null` for the
full eligible cohort once timed), stronger models (Med42-70B as primary
and five more, vs `llama3.1:8b`), concurrent LLM requests (`performance.llm_max_concurrent_requests`,
which the jobs also pass to Ollama as `OLLAMA_NUM_PARALLEL`;
`performance.ollama_flash_attention` sets `OLLAMA_FLASH_ATTENTION`), a
GPU-batched and larger LSTM baseline, checkpointing, and explicit DuckDB
thread/memory limits matching step 2's Slurm request. `jobs/bench_ollama.sh`
measures Ollama settings before you change them. The `#SBATCH` CPU/memory
requests (12 cores per GPU job) are a conservative one-GPU share of a node
(Rorqual's per-GPU bundle is 16 cores, 124 GB); check your cluster's with
`sinfo`. On Trillium every job gets a quarter node.

None of this unlocks the excluded full-scale (DT-GPT-size, ~35,000-patient)
cloud replication: that design calls a third-party-hosted model over the
network, which the dissertation's local-only constraint rules out
regardless of hardware.

### Project layout

```
logs/                                     job .out files and Ollama logs by default (paths.logs_dir; git-ignored); <run>/ per run_all.sh run name
tuned_configs/<run>/                      run_all.sh's per-run config and tuned config (git-ignored)
jobs/setup_bash.sh                        reads your profile: locations, account, symlinks, .venv
jobs/load_profile.sh                      sourced by every job: --profile, setup_bash.sh, PhysioNet credentials
jobs/run_all.sh lean|full                 the whole pipeline as chained Slurm jobs (login node)
jobs/pack_run.sh <run_name>               one run's logs, work dir and configs as <run_name>_results.tar
jobs/submit.sh [opts] <job> [args]        submit one job adapted to the cluster (logs_dir, gpu_jobs_only)
jobs/monitor-job.sh                       follow one job (login node)
jobs/smoke_test.sh                        pre-flight: synthetic data, mocked LLM
jobs/prep1_download_mimic.sh              download MIMIC-IV into $DT_MIMIC_DIR
jobs/prep2_download_models.sh <cfg>       download the Ollama models into $DT_OLLAMA_MODELS
jobs/bench_ollama.sh                      Ollama throughput benchmark
jobs/probe_output.sh <cfg> <variant>      what one model generates for the forecast prompt (schema vs plain JSON)
jobs/step1_resolve_items.sh <cfg>         step 1
jobs/step2_extract_cohort.sh <cfg>        step 2
jobs/step3b_tune.sh <cfg>                 step 3b (GPU)
jobs/step3c_calibrate.sh <tuned>          step 3c (GPU)
jobs/step3_run_experiment.sh <tuned>     step 3 (GPU)
jobs/step4_evaluate_results.sh <tuned>   step 4
jobs/ollama_lib.sh                        start/stop Ollama inside a GPU job
jobs/progress_lib.sh                      log-friendly download progress (prep1, prep2)
config/config_alliance_lean.yaml          5-variable panel (the default)
config/config_alliance_full.yaml          19-variable panel
config/tuning_grid.yaml                   hyperparameter search grid (step 3b)
config/profile.sample.yml                 template for your profile, ~/.config/dt_profile.yml
config/config_local.yaml                  original single-workstation (1080 Ti) config
src/common.py                 config loading, logging, path/DuckDB helpers
src/resolve_items.py          step 1 — itemid resolution
src/extract_cohort.py         step 2 — cohort + panel extraction (DuckDB)
src/tune.py                   step 3b — hyperparameter search
src/calibrate.py              step 3c — split-conformal interval calibration
src/run_experiment.py         step 3 — main entry point
src/evaluate_results.py       step 4 — RQ1/RQ2/RQ3 statistical analysis
src/select_rq_model.py        choose another model for the RQs (validation data)
src/harmonization_agent.py    data-harmonization (ETL) agent
src/similarity_agent.py       patient-similarity agent (local k-NN)
src/llm_client.py             local LLM wrapper (Ollama, 127.0.0.1 only)
src/forecasting_agent.py      LLM-based forecasting agent
src/critic_agent.py           critic / validation agent
src/baselines.py              naive, GBM, LSTM baselines
src/pipeline.py               orchestrates every condition
src/checkpoint.py             checkpoints and their fingerprints
src/metrics.py                evaluation metrics (sMAPE, KS, coverage, PVR, ...)
src/tracking.py               optional W&B metrics (aggregates only)
src/bench_ollama.py           Ollama benchmark (jobs/bench_ollama.sh)
src/probe_output.py           LLM output probe (jobs/probe_output.sh)
src/smoke_test.py             synthetic end-to-end test
docs/                         walkthroughs of the jobs' settings, steps 1 and 2, runtime estimates
```

### Notes and known limitations

- **Itemid resolution is pattern-based, not hardcoded** — deliberately,
  since hardcoded itemids are a common source of silent errors across
  MIMIC-IV point releases. Review `item_mapping.json` before trusting the
  extracted panel.
- **`anchor_age`** (used for the age ≥ 18 filter) is MIMIC-IV's
  de-identified, shifted age anchored to `anchor_year`, the standard proxy
  used across the MIMIC-IV literature — not necessarily the exact age at
  this admission.
- **The critic's correction call reuses the forecasting agent's model**
  (Chapter 5's design) unless a variant sets `critic_variant`. If a
  correction still leaves a value out of range after
  `critic_agent.max_correction_attempts`, it is clipped to the nearest
  bound, so a stubborn generation can never block the pipeline.
- **Missing variables (methodology).** When the model returns valid JSON
  but leaves some variables out, only those get the naive forecast (last
  observed value carried forward, interval half-width 1.5 × the
  observation-window SD); its other forecasts are kept.
  - Before this rule, any omission discarded the whole forecast: 58 in the
    first lean runs (jobs 23143356–23150606), almost all because lactate,
    the most sparsely measured variable, was left out.
  - Filled values are scored like any other forecast, and counted per
    condition and variable (`llm_filled_rate` in `all_conditions_summary.csv`,
    `filled_rate` in tuning's `trials.csv`, `llm_filled` in
    `calibration.json`); report them alongside the fallback rate.
  - Checkpoints made under the old rule aren't reused.
- **Ollama server errors are retried.** An HTTP 5xx is retried twice before
  the call counts as failed, and the error body is logged. Connection errors
  (the server is gone, e.g. at the job's time limit) stop the run instead of
  being recorded as fallbacks.
- **Forecast length is enforced while decoding.** Forecasts and critic
  corrections are requested with a JSON schema as Ollama's `format`, so each
  variable gets exactly `forecast_horizon_hours` values.
  - Before, with plain JSON mode, Qwen2.5-32B often ran on to 30–31 values
    and the whole forecast fell back to naive: 118 of 128 forecasts in the
    `similarity_context: trajectory` tuning trial (job 23206117).
  - The schema fixes the shape, not the content: runaway straight-line
    trends are now scored as the model's forecast instead of being hidden
    behind the fallback.
  - Checkpoints made before this change aren't reused.
