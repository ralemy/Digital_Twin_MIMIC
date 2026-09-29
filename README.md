# Local MIMIC-IV Digital Twin Experiment (Lean Scope)

Runnable implementation of Chapter 5's (lean-scope, fully local, no-cloud,
no-external-API) methods: a data-harmonization agent, a patient-similarity
agent, an LLM-based forecasting agent, a critic/validation agent, three
non-LLM baselines (naive, gradient-boosted trees, LSTM), and the RQ1–RQ3
statistical analysis, all running on your own machine against a local
MIMIC-IV export at `~/mimic-iv/{hosp,icu}`.

Originally verified against a GTX 1080 Ti / 32GB RAM machine, comfortably
sufficient for this lean-scope design — see the earlier discussion on
hardware feasibility. This is **not** the excluded, cloud-only full-scale
replication (Chapter 7) — see that discussion for why, and see "Scaling up
on better hardware" below if you've since moved to a bigger machine.

## 0. One-time setup

```bash
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a small quantized open-weight
# model — anything ~7-8B in 4/5-bit quantization comfortably fits an 11 GB card:
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve   # usually auto-starts as a background service; leave it running
```

Confirm your MIMIC-IV export is laid out as the standard PhysioNet download:

```
~/mimic-iv/
  hosp/   patients.csv(.gz)  admissions.csv(.gz)  labevents.csv(.gz)  d_labitems.csv(.gz)  ...
  icu/    icustays.csv(.gz)  chartevents.csv(.gz)  d_items.csv(.gz)  inputevents.csv(.gz)  ...
```

Files can be either raw `.csv` or gzip-compressed `.csv.gz` — the code
handles both. Nothing is uploaded anywhere; every step below runs against
the files on disk and, for the LLM calls, against `127.0.0.1:11434` only.

## 1. Confirm the code runs before touching real data (recommended)

```bash
python src/smoke_test.py
```

Builds a small synthetic cohort, mocks the LLM, and runs every condition end
to end in under a minute. If this fails, something in your environment
(missing dependency, Ollama not running) needs fixing before step 2–4 will
work — much cheaper to find out now than after a multi-hour extraction.

## 2. Resolve the variable panel to real itemids

```bash
python src/resolve_items.py --config config/config.yaml
```

Writes `~/mimic-iv-twin-work/cache/item_mapping.json`. **Open and read this
file before continuing** — it lists every `d_items`/`d_labitems` row matched
for each of the five panel variables (heart rate, resp rate, SpO2, MAP,
lactate). MIMIC-IV changes itemids across releases and units can differ
across care units (e.g. arterial vs. non-invasive MAP); this step resolves
by label pattern against your actual files rather than hardcoding ids, but
you should still eyeball it once. Edit the JSON directly if you want to
drop or add an itemid.

## 3. Extract the cohort and panel

```bash
python src/extract_cohort.py --config config/config.yaml
```

This is the slow step on first run: it does one filtered pass each over
`chartevents` and `labevents` (the two largest MIMIC-IV tables) and caches
the result as Parquet, so every subsequent run is fast. Expect this to take
anywhere from several minutes to an hour or more on first run depending on
disk speed — it is not using your GPU. Progress is logged.

Writes `~/mimic-iv-twin-work/cohort.parquet` and `panel_long.parquet`.
`config.yaml`'s `cohort.max_patients: 300` subsamples to a fast dev-sized
cohort by default — set it to `null` to use the full eligible cohort once
you've confirmed everything else works.

## 4. Run the experiment

```bash
python src/run_experiment.py --config config/config.yaml
```

Fits the GBM and LSTM baselines, builds the similarity index, and runs every
condition listed in `config.yaml` (`naive`, `gbm`, `lstm`, `single_model_llm`,
`full_pipeline`, and the two orchestration ablations) over the held-out test
split. This is the step that calls your local LLM repeatedly — with the
default 300-patient dev cohort this should finish in well under an hour on
the 1080 Ti; scale up `max_patients` once you've checked the results look
sane. Raw forecast arrays and per-condition metric summaries are written to
`~/mimic-iv-twin-work/results/`.

## 5. Statistical analysis

```bash
python src/evaluate_results.py --config config/config.yaml
```

Produces the RQ1 (orchestration vs. single-model), RQ2 (critic ablation:
plausibility-violation-rate reduction + accuracy trade-off check), and RQ3
(stable vs. deteriorating subgroup) analyses described in Chapter 5, Section
5.6, with bootstrap 95% confidence intervals computed by resampling patients.
Writes `~/mimic-iv-twin-work/results/statistical_analysis.json`.

## Scaling up on better hardware

If you've moved from the original reference machine (11GB GTX 1080 Ti,
32GB RAM) to something bigger — e.g. a ~40GB GPU, ~400GB RAM, 24+ CPU
cores — everything above still works unchanged, but leaves most of that
hardware idle. Use `config/config_highend.yaml` instead of `config.yaml`
for every command above (`--config config/config_highend.yaml`) to
actually use it. It's the same design — same 5-variable panel, same
cohort criteria, same conditions/ablations — just sized differently:

- **Larger (or full) cohort.** `cohort.max_patients` was capped at 300
  mainly to keep sequential LLM inference wall-clock time reasonable, not
  because extraction or the baselines needed it. The high-end config
  raises it to 3000 as a starting point; set it to `null` for the full
  eligible cohort once you've timed one run.
- **A stronger local model.** An 11GB card only fits a small quantized
  model (8B/q4). A 40GB card fits something much more capable — the
  high-end config defaults to `qwen2.5:32b-instruct-q8_0` (~34GB weights,
  leaves headroom for context/KV cache); `llama3.1:70b-instruct-q4_K_M`
  (~40GB) is a documented alternative in that file if you want to try it,
  but it leaves little headroom once you raise `num_ctx` or run more than
  one request at a time — benchmark one condition before committing to a
  full run with it. Pull whichever you use with `ollama pull <name>`
  before running step 4.
- **Concurrent LLM requests.** The forecasting/critic agents used to call
  Ollama once per patient, sequentially. `src/pipeline.py` now runs those
  calls through a thread pool sized by `performance.llm_max_concurrent_requests`.
  This only helps if Ollama itself is willing to process that many
  requests at once — start it with, e.g.:
  ```bash
  OLLAMA_NUM_PARALLEL=4 ollama serve
  ```
  matching (or exceeding) `llm_max_concurrent_requests` in the config, and
  watch `nvidia-smi` on your first run — if VRAM is tight, lower one or
  both numbers rather than letting Ollama OOM mid-run.
- **GPU-batched LSTM baseline.** With `performance.batch_predict_baselines:
  true`, the LSTM baseline's `predict()` runs once over the whole test
  split in large GPU batches instead of once per patient — the per-patient
  loop barely used a 40GB card at all. `LSTMBaseline.fit()` was already
  full-batch (the whole training tensor stack in one shot), so it needed no
  change beyond `lstm_hidden_size`/`lstm_epochs` now being configurable and
  turned up (128 hidden units, 200 epochs vs. 64/30) since training is
  seconds either way on this hardware.
- **DuckDB given its own thread/memory budget.** `performance.duckdb_threads`
  and `performance.duckdb_memory_limit_gb` (used by every DuckDB connection
  via `src/common.py`'s `get_duckdb_connection()`) — set explicitly because
  DuckDB's auto-detection can undershoot what's actually available inside a
  container or restricted cgroup. Extraction (step 3) is CPU/RAM-bound, not
  GPU-bound, so this is what actually speeds that step up, not the GPU.

**What this does *not* unlock: the excluded full-scale (DT-GPT-size,
~35,000-patient) cloud replication described in the proposal's Cost
chapter.** That design is excluded because it calls a third-party-hosted
managed model over the network, which the dissertation's local-only
constraint rules out regardless of local hardware — it was never a
question of VRAM, RAM, or CPU count. This hardware upgrade lets the
*lean-scope* design run at a much larger local cohort and with a much
stronger local model, faster — it doesn't change which design is in
scope.

**What this also does not do on its own: add more variables to the
panel.** The five-variable panel (heart rate, respiratory rate, SpO2,
MAP, lactate) was chosen for clinical relevance and coverage, not because
the 1080 Ti couldn't handle more variables — extraction and inference
cost scale roughly linearly with variable count and were never the
binding constraint. `config_highend.yaml` deliberately keeps the same
five variables, because that panel is also what's already been described
to UVic's HREB (Chapter 5/Appendix A, and the Appendix A data collection
sheet in the letter to the Board). Expanding it is a scope decision, not
a hardware one — see the next section for the alternate config that does
expand it, and why it is kept separate.

## Two configs, two scopes — pick deliberately

There are now five config files, not one, and they are **not**
interchangeable:

| File | Panel | Hardware | Matches current HREB approval? |
|---|---|---|---|
| `config/config.yaml` | 5 variables | 1080 Ti / 32GB (small dev cohort) | Yes |
| `config/config_highend.yaml` | 5 variables (same panel) | 40GB GPU / 400GB RAM / 24 CPU (local machine) | **Yes** — same design UVic's HREB has seen, just faster/larger cohort |
| `config/config_highend_full_variables.yaml` | 19 variables | same high-end local hardware sizing | **No** — see below |
| `config/config_nibi_lean.yaml` | 5 variables (same panel) | Nibi, 1× H100 80GB Slurm job | **Yes** — same design as `config_highend.yaml`, different infrastructure |
| `config/config_nibi_full_variables.yaml` | 19 variables | Nibi, 1× H100 80GB Slurm job | **No** — same caveat as `config_highend_full_variables.yaml` |

`config_highend_full_variables.yaml` adds 3 more vitals (systolic/diastolic
blood pressure, temperature) and 11 more labs (creatinine, BUN, sodium,
potassium, WBC, platelets, hemoglobin, bicarbonate, glucose, total
bilirubin, pH) to the same 5-variable core, for a 19-variable panel. This
is a proposed expansion, not a panel drawn from any existing chapter or
appendix — review and adjust `variables:` in that file before treating it
as final. Nothing about *how* the code works changes for a bigger panel
(`resolve_items.py`, `extract_cohort.py`, the harmonization/similarity/
forecasting/critic agents, and the baselines all iterate over
`cfg["variables"]` generically — this was true even at 5 variables), so
no source files needed to change, only the config. It writes to a
separate `work_dir` (`~/mimic-iv-twin-work-full`) so it never overwrites
the 5-variable extraction/results.

**Do not run `config_highend_full_variables.yaml` against real MIMIC-IV
data yet.** The methods text (Chapter 5) and the letter already sent to
UVic's HREB (`Letter_to_UVic_HREB.docx`, Appendix A) both describe the
5-variable panel specifically — the letter states in writing that "this
matches the version of the variable panel the Board is seeing, and will
not change." Running the 19-variable config produces results outside
what's currently approved. It's included here so the code is ready
*if and when* a deliberate decision is made to expand the panel and an
amendment is submitted to the Board — as of this version, that decision
has not been made and no such amendment has been prepared or sent.

## Running on Nibi (Alliance Canada HPC)

`config_nibi_lean.yaml` and `config_nibi_full_variables.yaml` are the same
two scopes described above (5 variables / matches HREB approval, vs. 19
variables / does not), retargeted from a local high-end machine to a
[Nibi](https://docs.alliancecan.ca/wiki/Nibi) Slurm allocation — one H100
80GB GPU per job. Nothing about the pipeline code changed for this; only
paths (`$PROJECT`/`$SCRATCH` instead of `~/...`) and the CPU/thread numbers
(capped to Nibi's documented "no more than 14 CPU cores per GPU" guidance,
not the 24-core local machine's number) differ from `config_highend*.yaml`.
`src/common.py`'s `load_config()` expands `$ENV_VARS` as well as `~` in
every `paths:` entry, so these configs only resolve correctly inside an
Alliance job/login environment where `$PROJECT`/`$SCRATCH` are set.

There are five submission scripts under `jobs/`, one for the smoke test and
one per pipeline step, rather than a single job that runs everything —
this way a failed or resubmitted step doesn't re-run (and re-burn GPU
allocation on) steps that already succeeded, and each step gets its own
right-sized `#SBATCH` resources instead of one job requesting the maximum
any step needs. All four step scripts take the config file as their first
argument, so the same script runs either scope — omit it and it defaults
to `config/config_nibi_lean.yaml` (the HREB-approved scope):

```bash
# Set once per shell (sbatch passes these through to every job below) —
# see "Paths the job scripts need" further down for what each one is:
export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
export OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama_local/models

# 0. Pre-flight — no GPU, no Ollama, no real data needed:
sbatch jobs/smoke_test_nibi.sh

# 1-4. The real pipeline, lean scope (HREB-approved):
sbatch jobs/step1_resolve_items_nibi.sh   config/config_nibi_lean.yaml
sbatch jobs/step2_extract_cohort_nibi.sh  config/config_nibi_lean.yaml
sbatch jobs/step3_run_experiment_nibi.sh  config/config_nibi_lean.yaml   # GPU + Ollama
sbatch jobs/step4_evaluate_results_nibi.sh config/config_nibi_lean.yaml

# Same four scripts for the alternate (19-variable, NOT yet HREB-approved) scope:
sbatch jobs/step1_resolve_items_nibi.sh   config/config_nibi_full_variables.yaml
sbatch jobs/step2_extract_cohort_nibi.sh  config/config_nibi_full_variables.yaml
sbatch jobs/step3_run_experiment_nibi.sh  config/config_nibi_full_variables.yaml
sbatch jobs/step4_evaluate_results_nibi.sh config/config_nibi_full_variables.yaml

sq   # check job status
```

Each step depends on the previous one's output on disk (`item_mapping.json`,
then the cohort/panel parquet files, then the `*_raw.npz` results), so
submit them one at a time and confirm each finished, or chain them so Slurm
only starts the next once the previous succeeds:

```bash
J1=$(sbatch --parsable jobs/step1_resolve_items_nibi.sh  config/config_nibi_lean.yaml)
J2=$(sbatch --parsable --dependency=afterok:$J1 jobs/step2_extract_cohort_nibi.sh  config/config_nibi_lean.yaml)
J3=$(sbatch --parsable --dependency=afterok:$J2 jobs/step3_run_experiment_nibi.sh  config/config_nibi_lean.yaml)
J4=$(sbatch --parsable --dependency=afterok:$J3 jobs/step4_evaluate_results_nibi.sh config/config_nibi_lean.yaml)
```

Only step 3 (`run_experiment.py`) touches the GPU or Ollama — it requests
`--gpus-per-node=h100:1`, starts `ollama serve` in the background on the
compute node (bound to `127.0.0.1`, matching the local-only constraint),
waits for it to become ready with a `curl`-based polling loop, and stops it
in a `trap ... EXIT` cleanup handler so the server is killed whether the
run succeeds or fails. Steps 1, 2 and 4 request CPU/RAM only and finish
much faster. `--cpus-per-task`/`--mem` on each script are sized to that
step's actual work (step 2's extraction is the heaviest, matching the
config's `duckdb_threads`/`duckdb_memory_limit_gb`) — there was no
confirmed Nibi per-node spec sheet available while preparing these, only
the public "no more than 14 cores per GPU" ratio, so check
`sinfo -o "%N %c %m %G"` once you have a session and raise the `#SBATCH`
lines if more is actually available.

**Paths the job scripts need, read from the environment.** All five
`jobs/*.sh` scripts read the project directory from `$AGENTIC_DT_PRJ` (and
`jobs/step3_run_experiment_nibi.sh` additionally reads the Ollama model
store from `$OLLAMA_MODELS` — the standard Ollama env var name, so the same
export also works for a manual `ollama pull`/`ollama serve`). Neither is
hardcoded in the scripts, so if this codebase or the model store ever move,
nothing in `jobs/` needs editing — just export the new values before
`sbatch`. Each script fails fast with a clear message if its required
variable isn't set, rather than silently doing the wrong thing. `sbatch`
passes the submitting shell's environment through by default, so exporting
these once per shell session (or once in `~/.bashrc`) is enough:

```bash
export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
export OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama_local/models
```

Within `$AGENTIC_DT_PRJ`, the scripts expect:

- `.venv` — the Python virtualenv every script activates
  (`source "$AGENTIC_DT_PRJ/.venv/bin/activate"`)
- `src/`, `config/` — this repository's `src/` and `config/` as checked out,
  unchanged

`$OLLAMA_MODELS` is deliberately a separate variable rather than a path
under `$AGENTIC_DT_PRJ`, so the multi-GB model weights aren't duplicated
per experiment directory if you ever have more than one.

**One-time setup, before the first `sbatch`** (all five scripts share this;
do it with both exports above already set in your shell):

1. **Build the Python venv once, on a login node, inside `$AGENTIC_DT_PRJ`**
   — compute nodes typically have no internet access, so installing
   packages has to happen where a network path to PyPI/the Alliance wheel
   mirror exists:
   ```bash
   module load python/3.11
   cd "$AGENTIC_DT_PRJ"
   virtualenv --no-download .venv
   source .venv/bin/activate
   pip install --no-index --upgrade pip
   pip install --no-index -r requirements.txt
   deactivate
   ```
   If a package isn't available in the Alliance wheel mirror, drop
   `--no-index` for that one `pip install` only.
2. **Pull the Ollama model once, on a login node**, for the same reason —
   `jobs/step3_run_experiment_nibi.sh` assumes the model is already present
   under `$OLLAMA_MODELS` and aborts with a clear message rather than
   trying (and failing) to pull it mid-job:
   ```bash
   ollama pull qwen2.5:32b-instruct-q8_0
   ```
3. **Place the MIMIC-IV export under `$PROJECT/mimic-iv/{hosp,icu}`**
   (`config_nibi_*.yaml` resolve `mimic_root`/`work_dir` via the `$PROJECT`
   env var Alliance itself sets in the job environment — a different
   variable from `$AGENTIC_DT_PRJ` above), not `$HOME` (50GB quota, and not
   persistent in the way this needs) — `$PROJECT` persists (unlike
   `$SCRATCH`, purged after 60 days of inactivity). Before uploading
   anything, confirm Nibi's storage meets PhysioNet's Data Use Agreement
   requirements for where credentialed MIMIC-IV data may be stored — that
   check is on you, the
   config/job files don't and can't verify it.

Both `config_nibi_*.yaml` files carry the same header warnings inline;
read them before your first real submission, not just this section.

## Project layout

```
config/config.yaml                       5-variable panel, 1080 Ti-sized (matches current HREB approval)
config/config_highend.yaml               same 5-variable panel, sized for bigger local hardware (matches current HREB approval)
config/config_highend_full_variables.yaml 19-variable panel, same local hardware sizing (NOT yet covered by HREB approval — see above)
config/config_nibi_lean.yaml             same 5-variable panel, sized for a Nibi Slurm job (matches current HREB approval)
config/config_nibi_full_variables.yaml   19-variable panel, sized for a Nibi Slurm job (NOT yet covered by HREB approval — see above)
jobs/smoke_test_nibi.sh                  sbatch script — pre-flight smoke test, no GPU/Ollama/real data
jobs/step1_resolve_items_nibi.sh <cfg>   sbatch script — pipeline step 1, either scope
jobs/step2_extract_cohort_nibi.sh <cfg>  sbatch script — pipeline step 2, either scope
jobs/step3_run_experiment_nibi.sh <cfg>  sbatch script — pipeline step 3 (GPU + Ollama), either scope
jobs/step4_evaluate_results_nibi.sh <cfg> sbatch script — pipeline step 4, either scope
src/common.py              config loading, logging, path/DuckDB-connection helpers
src/resolve_items.py       Step 2 — itemid resolution
src/extract_cohort.py      Step 3 — cohort + panel extraction (DuckDB, local only)
src/harmonization_agent.py data-harmonization (ETL) agent
src/similarity_agent.py    patient-similarity agent (local k-NN)
src/llm_client.py          local LLM wrapper (Ollama, 127.0.0.1 only)
src/forecasting_agent.py   LLM-based forecasting agent
src/critic_agent.py        critic / validation agent
src/baselines.py           naive, GBM, LSTM baselines
src/pipeline.py            orchestrates every condition
src/metrics.py             evaluation metrics (sMAPE, KS, coverage, PVR, ...)
src/run_experiment.py      Step 4 — main entry point
src/evaluate_results.py    Step 5 — RQ1/RQ2/RQ3 statistical analysis
src/smoke_test.py          synthetic end-to-end test, no real data / no Ollama needed
```

## Notes and known limitations of this reference implementation

- **Itemid resolution is pattern-based, not hardcoded** — deliberately, since
  hardcoded itemids are a common source of silent errors across MIMIC-IV
  point releases. Review `item_mapping.json` before trusting the extracted
  panel.
- **`anchor_age`** (used for the age >= 18 filter) is MIMIC-IV's de-identified,
  shifted age field anchored to `anchor_year`, the standard proxy for age
  used across the MIMIC-IV literature — not necessarily the patient's exact
  age at this specific admission.
- **The LSTM baseline needs PyTorch.** If you'd rather not install it, set
  `baselines.run_lstm: false` and remove `lstm` from `conditions` in
  `config.yaml`; everything else runs without it.
- **The critic agent's correction call reuses the same local model** as the
  forecasting agent (per Chapter 5's design — no larger or externally-hosted
  model is used anywhere). If a correction still leaves a value out of range
  after `critic_agent.max_correction_attempts`, it is clipped to the nearest
  bound rather than retried indefinitely, so a stubborn generation can never
  block the pipeline.
- **This is a reference implementation, not a tuned one.** GBM/LSTM
  hyperparameters, the LLM prompt wording, and the similarity feature set are
  all reasonable starting points, not the result of hyperparameter search —
  expect to iterate once you see results on your real cohort.
