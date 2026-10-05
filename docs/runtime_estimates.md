# Run-time estimates — lean (5-variable) and full (19-variable) scopes

How long it takes, on Nibi with one H100, to go from the extracted cohort to
the RQ1–RQ3 answers: hyperparameter search (`tune.py`), interval calibration
(`calibrate.py`), the final experiment with the tuned settings
(`run_experiment.py`), and the statistical analysis (`evaluate_results.py`).

The lean numbers are **measured** from the job logs in the project directory
(Oct 2–3, 2026). The full-variable numbers are **extrapolated**: no
19-variable job has been run yet. The section on the full scope explains how
they were scaled.

Times are GPU-hours on one H100 80GB, with `OLLAMA_NUM_PARALLEL=4` and
`checkpoint_batch_size=32`. Queue waits between chained jobs were only a few
minutes in these logs, so wall-clock time is close to GPU time when jobs are
chained with `--dependency=afterany`.

---

## 1. What the logs measured

### Steps 1–2 (CPU)

| Job | Step | Duration |
|---|---|---|
| 23138952 | `resolve_items.py` | 6 s |
| 23139063 | `extract_cohort.py` (3,000 of 35,394 eligible stays) | 4 min |

Both steps are negligible compared with the LLM steps.

### Step 3 — the untuned lean experiment (450 test patients, 15 batches)

Jobs 23143356, 23150580 and 23150606 (config `config_alliance_lean.yaml`):

| Condition | Model | Wall time | Per batch of 32 |
|---|---|---|---|
| naive / gbm / lstm | — | ~1 min total | — |
| `single_model_llm` | qwen2.5:32b | ~77 min | ~5.1–5.3 min |
| `full_pipeline` | qwen2.5:32b | 84 min | 5.6 min |
| `full_pipeline_no_critic` | qwen2.5:32b | ~75 min | 5.0 min |
| `full_pipeline_no_similarity` | qwen2.5:32b | 78 min | 5.2 min |
| `single_model_llm@gemma3` | gemma3:27b | 76 min | 5.1 min |
| `full_pipeline@gemma3` | gemma3:27b | 84 min | 5.6 min |
| `single_model_llm@medgemma` | medgemma:27b | 76 min | 5.0 min |
| `full_pipeline@medgemma` | medgemma:27b | 84 min | 5.6 min |
| `single_model_llm@llama_Med42_70b` | med42:70b | ~80 min | 5.3 min |
| `full_pipeline@llama_Med42_70b` | med42:70b | 83 min | 5.5 min |
| **All 10 LLM conditions** | | **≈ 13.3 h** | |

What these numbers show:

- **About 0.17 min per patient for a single-model forecast, and about 0.19 min
  for the full pipeline.** The critic and similarity agents add roughly 10%.
- **Model size barely matters.** The 70B Med42 (Q4) runs at the same speed as
  the 27–32B models. All four generate at about 16–21 tokens/s per request
  slot (Ollama `print_timing` lines), so about 70–80 tokens/s across the
  4 slots.
- **Run time is set by the number of tokens generated, not by prompt length.**
  The median forecast call takes 38 s (p10 8 s, which are the short critic
  calls; p90 42 s; max 161 s). In job 23141406, `max_tokens=600` cut off every
  forecast and batches took about 3.2 min. With `max_tokens=1536` and complete
  forecasts of about 700–900 tokens, they take about 5.2 min. This is the
  basis for scaling to the 19-variable panel.
- **One 8-hour job can't hold a full run.** Job 23150580 hit its time limit
  during the 2nd Med42 batch. Resuming from checkpoints worked: job 23150606
  finished the run, losing only the batch that was in progress.

### Checkpoint integrity

Job 23143356 reported a 31.8% fallback rate for `single_model_llm` because it
reused batches that job 23141406 had marked "done" while being cancelled. In
those batches every Ollama call had failed with a connection error. Those
batches have since been recomputed. The current checkpoints show 20/450
fallbacks (4.4%), all from runs with `max_tokens=1536`. The forecasting agent
now re-raises `ConnectionError` instead of recording a naive forecast, so this
can't happen again.

---

## 2. Cost model

| Quantity | Lean (5 var) | Full (19 var, estimated) |
|---|---|---|
| Forecast JSON length | ~700–900 tokens | ~2,500–3,500 tokens (config comment) |
| Per batch of 32, single-model condition | 5.2 min | ~17–22 min |
| Per batch of 32, full-pipeline condition | 5.6 min | ~19–24 min |
| Factor applied to LLM time | 1× | **3.7× (range 3.3–4.2×)** |

The full-scope factor is the ratio of output tokens. The longer prompt
(19-variable observation summary, about 1.5k tokens) adds little, because
prompt processing is a small share of each call. The critic may make more
calls with 19 variables, but critic calls are short (~8 s), so they're
covered by the upper end of the range.

The full-scope figures assume the cohort is still capped at
`max_patients: 3000`, giving 450 test and 450 validation patients
(128 tuning + 322 calibration). With 19 variables, fewer stays pass the 50%
coverage filter. If fewer than 3,000 remain, every LLM time below shrinks in
proportion. The extraction log reports the eligible count.

### The three LLM-bound steps, in lean units

- **Tuning** (`config/tuning_grid.yaml`, 128 patients = 4 batches, qwen2.5
  only). Settings already evaluated are cached and not re-run, which leaves
  14 new LLM evaluations:
  - `similarity_context`: 2 × full_pipeline
  - `k_neighbors`: 2 × full_pipeline
  - `prompt`: 4 × single + 3 × full
  - `temperature`: 1 × single + 1 × full
  - `critic`: 1 × full

  Total: 5 × ~21 min + 9 × ~23 min ≈ **5.2 h**, plus about 10 min of
  GBM/LSTM trials and model loading.
- **Calibration** (every condition in the config, 322 patients ≈ 10.1
  batches). The 10 LLM conditions take about 53 min per batch round in
  total, so about **9 h**.
- **Final run with tuned settings** (every condition, 450 test patients). It
  writes to its own `results_tuned` / `checkpoints_tuned`, so none of the
  untuned run is reused: about **13.3 h**. If the winning settings include
  `trajectory` context, `k=50` or 2 critic attempts, add about 5–15%. If
  `recent_hours: 6` wins, the prompt gets shorter and the run is slightly
  faster.
- **Evaluation** (`step4`, CPU, 2,000 bootstrap resamples): under 30 min.

---

## 3. Estimates

### Lean scope (5 variables)

| Step | GPU-h | 8-h jobs | Notes |
|---|---|---|---|
| Steps 1–2 | ~0.1 (CPU) | 1 each | |
| Untuned experiment (optional) | ~13.5 | 2 | A "before tuning" reference in `results/`; `run_all.sh` skips it |
| Tuning (`step3b`) | ~5.5 | 1 | Within the 8 h limit |
| Calibration (`step3c`) | ~9 | 2 | Chain a second job with `afterany` |
| Final tuned run (`step3`) | ~13.5–15 | 2 | |
| Evaluation (`step4`) | <0.5 (CPU) | 1 | |
| **Total, without the untuned run** | **~28–30** | **8** | **About 1.5 days** of wall-clock time if the jobs are chained |

### Full scope (19 variables)

Extrapolated, not measured (see section 2). As for the lean scope, the
approvals and permissions in the README's "Before you start" section must
be in place — and must cover the 19-variable panel — before
[`config_alliance_full.yaml`](../config/config_alliance_full.yaml) is run
against real MIMIC-IV data.

| Step | GPU-h (range) | 8-h jobs | Notes |
|---|---|---|---|
| Steps 1–2 | ~0.2 (CPU) | 1 each | More item IDs, same scan |
| Tuning, full grid | ~19 (17–22) | 3 | |
| Tuning, cut-down grid¹ | ~12 (11–14) | 2 | |
| Tuning, reuse lean settings² | 0 | 0 | |
| Calibration | ~33 (30–38) | 5 | |
| Final tuned run | ~49 (44–56) | 7 | |
| Evaluation | <0.5 (CPU) | 1 | |
| **Total, full grid** | **~102 (92–117)** | **~16** | **About 4.5–5 days** if the jobs are chained |
| **Total, reusing lean settings** | **~83 (75–95)** | **~13** | **About 3.5–4 days** |

¹ The cut-down grid keeps the `k_neighbors` and `prompt` rounds plus the cheap
GBM/LSTM rounds. These are the settings most likely to change with a 19-dimensional
similarity space and a longer prompt.
² Apply `results/tuning/tuned_overrides.yaml` from the lean run on top of
`config_alliance_full.yaml`. Calibration still has to be redone: the
factors are fitted per variable, and `evaluate_results.py` rejects factors
whose fingerprint (including the variable panel) doesn't match.

### Cheaper paths to the RQ answers

RQ1–RQ3 need only some of the 10 LLM conditions:

| RQ | Comparison | Conditions |
|---|---|---|
| RQ1 | Orchestration vs single model | `single_model_llm`, `full_pipeline` |
| RQ2 | Critic ablation (plausibility + accuracy) | `full_pipeline`, `full_pipeline_no_critic` |
| RQ3 | Stable vs deteriorating subgroups | the RQ1/RQ2 conditions, stratified |

`full_pipeline_no_similarity` (similarity ablation) and the six `@variant`
conditions (model comparison) are secondary analyses. Running only the three
RQ conditions plus the baselines for calibration and the final run cuts those
two steps to about 30% of their cost:

| Scope | GPU-h, RQ conditions only (tuning + calibration + final run) |
|---|---|
| Lean | ~5.5 + 2.8 + 4.2 ≈ **12.5** |
| Full, reusing lean settings | ~10 + 15 ≈ **25** |
| Full, cut-down grid | ~12 + 10 + 15 ≈ **37** |

The variants can be added later. Each condition is checkpointed on its own,
so adding lines to `conditions:` later only computes the new conditions.

### Throughput improvements not measured yet

- **`OLLAMA_NUM_PARALLEL=8` (with `llm_max_concurrent_requests: 8`).** Decoding
  is limited by memory bandwidth. With qwen2.5 loaded, Ollama reported
  37 GiB of VRAM free, so twice as many request slots fit for the 27–32B
  models. Expected gain: about 1.5–1.8× on the LLM steps.
  - Med42-70B at `num_ctx=12288` would need about 73 GB, which is too close
    to the 80 GB card. Keep it at 4–6 slots.
  - Changing the concurrency doesn't change the fingerprints, so existing
    checkpoints stay valid.
  - Test with one short job before relying on it.
- **`OLLAMA_FLASH_ATTENTION=1`.** It is off in the logs. Turning it on lowers
  KV-cache memory and helps more as contexts get longer, which matters for
  the full scope.

---

## 4. Changes made in response to the logs

These changes were made after the runs above. Every LLM condition's
fingerprint now differs from those runs, because of the new missing-variable
rule and the lean `max_tokens`. The next lean run will therefore recompute
its LLM conditions rather than reuse the old checkpoints. The baselines (naive,
GBM, LSTM) are unaffected.

1. **Ollama HTTP 500s are retried and their cause is logged.**
   - In job 23150606, 61 of Med42's 80 fallbacks were HTTP 500s, and
     Ollama's log gave no cause.
   - `LocalLLM.generate` now retries a 5xx twice and logs the response body.
     If all three attempts fail, the error message includes the body.
   - [src/llm_client.py](../src/llm_client.py)
2. **Shape failures show what the model returned.**
   - A "missing top-level keys" error now lists the keys the model did
     return. Every shape error also logs the first 300 characters of the
     output. This is for diagnosing MedGemma's 141 failures (10.4% single,
     20.9% full pipeline).
   - [src/forecasting_agent.py](../src/forecasting_agent.py)
3. **The 19-variable configs have a longer request timeout.**
   `request_timeout_s` is now 600 (was 240) in `config_alliance_full.yaml`.
4. **A missing variable gets the naive forecast for that variable only**,
   instead of the whole forecast falling back. This is a methods change. It
   is described in the README under "Notes and known limitations → Missing variables" and reported
   as `llm_filled_rate` per condition and variable. It would have rescued 58
   forecasts in these logs, almost all for lactate.
5. **`max_tokens` is 2048 (was 1536)** in the three 5-variable configs.
   This fixes the 9 truncated forecasts in these logs. The run-time estimates
   don't change: normal forecasts are still ~700–900 tokens, and only the
   rare runaway forecast gets longer before stopping.
6. **`jobs/step3_run_experiment.sh` is updated.**
   - It now starts Ollama through `jobs/ollama_lib.sh`, which uses a per-job
     port and waits with curl retries instead of a `sleep` loop.
   - It checks models with `require_models`.
   - Its header now gives the correct number of chained 8-hour jobs per scope.

7. **`jobs/run_all.sh` runs every stage end to end** (see the README,
   section 6).
   - It sizes each job from the estimates in section 3: at most 7 h of work
     per job, time limit = work + 1 h, capped at 8 h.
   - Within a stage, it chains continuation jobs with `afternotok`
     dependencies.
   - It can be re-attached to after a dropped connection.
   - Step 1's and step 4's default time limits are now 1:15 and 1:30
     (were 0:30), so each is at least 1 h more than its estimate.

### Still to decide

- **MedGemma's output shape.** Once the next MedGemma run logs what it
  returns, decide whether to accept its format or to treat it as a failure.
- **`OLLAMA_NUM_PARALLEL=8` / `OLLAMA_FLASH_ATTENTION=1`: adopted.** The
  benchmark (job 23198298: qwen2.5:32b, `full_pipeline`, 64 validation
  patients) measured:

  | Parallel requests | Flash attention | min per 450 patients | Speed-up | Fallbacks | sMAPE |
  |---|---|---|---|---|---|
  | 4 | off | 76.1 | 1.00× | 3/64 | 0.1773 |
  | 4 | on | 63.3 | 1.20× | 3/64 | 0.1769 |
  | 8 | off | 58.7 | 1.30× | 2/64 | 0.1775 |
  | 8 | on | 44.9 | 1.69× | 3/64 | 0.1782 |

  - At 8 slots with flash attention, Med42-70B (63.6 GB) and qwen2.5 at
    12K context (60.2 GB) fit entirely on the GPU.
  - The Alliance configs now set `llm_max_concurrent_requests: 8` and
    `ollama_flash_attention: true`.
  - `EST_*` in `jobs/run_all.sh` assume 1.69× for the 27–32B models and
    1.3× for the 70B models, whose speed at 8 slots wasn't measured.
  - With the 16 LLM conditions now in the configs, a full lean run is
    about 30 hours of estimated work (the 19-variable panel about 100).
