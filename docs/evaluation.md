# Evaluation — how this project judges an experiment

This page explains how the experiment's results are evaluated, why each
metric was chosen (and others weren't), what the first full lean run on
Trillium taught us and what we changed because of it, and how the
hyperparameters are tuned (section 4). Every number comes from that run, or from re-analysing
its saved forecasts. The run's full write-up is in its results folder
(`README.md` next to `mimic-iv-twin-work/`).

Code: `src/metrics.py` (the metrics), `src/run_experiment.py` (per-condition
metrics), `src/calibrate.py` (interval calibration), `src/evaluate_results.py`
(the statistical tests), `src/tune.py` (tuning).

---

## 1. How an experiment is evaluated

### 1.1 The forecasting task

Each ICU stay contributes one example. The first **24 hours** after admission
are the observation window. The model forecasts the next **24 hours**, hour by
hour, for every variable in the panel. The lean scope has 5 variables: heart
rate, respiratory rate, SpO2, MAP and lactate. The full scope has 19. The
ground truth is the hourly value MIMIC-IV recorded. Short gaps (≤ 3 h) are
interpolated, and hours with no measurement are left empty (`NaN`) and never
scored.

### 1.2 Patients, splits and what each split is used for

The cohort is adults with an ICU stay of at least 48 h. Only one stay per
patient is used (`first_stay_only`), so no patient appears in two splits. The
lean scope draws 3,000 stays, split at random (seed 42) 70 / 15 / 15:

| Split | Stays | Used for | Never used for |
|---|---|---|---|
| train | 2,100 | fitting GBM and LSTM; the similarity agent's neighbour index; the cohort medians that fill unobserved variables | scoring |
| validation, tuning part | 128 | `tune.py`: choosing hyperparameters and prompt options | the reported results |
| validation, calibration part | 322 | `calibrate.py`: interval calibration factors | choosing settings |
| test | 450 | the reported results and every statistical test | choosing anything |

Keeping tuning and calibration on separate validation patients means neither
the hyperparameters nor the interval factors have seen the test patients.

### 1.3 The conditions compared

Every condition forecasts the same 450 test patients, so all comparisons are
**paired**: two conditions are compared patient by patient.

- **Conventional baselines:** naive (last value carried forward), GBM (one
  gradient-boosted regressor per variable and forecast hour), LSTM
  (sequence-to-sequence).
- **LLM conditions:** `single_model_llm` (the LLM alone, DT-GPT style),
  `full_pipeline` (similarity agent + forecaster + critic), and three
  ablations of it: `_no_critic`, `_no_similarity` and `_clip_critic`. Each
  runs with the primary model (Qwen2.5-32B) and, as `@variant`, with the other
  local models.

### 1.4 The steps

1. **Tune** (`tune.py`, on the 128 tuning patients). A coordinate search: each
   round tries a few settings and keeps one only if it lowers the mean
   per-patient sMAPE by at least 0.002, with at most 10 % of LLM forecasts
   falling back to naive.
2. **Calibrate** (`calibrate.py`, on the 322 calibration patients). For each
   condition and variable, a split-conformal factor scales the forecast's
   interval so it covers about 90 % of outcomes (section 2.3).
3. **Run** (`run_experiment.py`, on the 450 test patients). Every condition's
   forecasts are saved (`<condition>_raw.npz`) and scored per variable
   (`all_conditions_summary.csv`, `<condition>_summary.txt`).
4. **Evaluate** (`evaluate_results.py`). Paired statistical tests answer the
   research questions and compare the models, and the calibration factors are
   applied to the test intervals (`statistical_analysis.json`).

### 1.5 The research questions and their tests

| Question | Comparison | Test |
|---|---|---|
| **RQ1** Does orchestration beat a single model? | `single_model_llm` vs `full_pipeline` | Wilcoxon signed-rank on per-patient sMAPE; bootstrap 95 % CIs |
| **RQ2** What does the critic do? | `full_pipeline_no_critic` vs `full_pipeline` (and `_clip_critic`) | Same accuracy test; plausibility-violation rates; what the critic corrected (section 3.3) |
| **RQ3** Does it hold for deteriorating patients? | stable vs deteriorating subgroups | Per-patient sMAPE per subgroup (descriptive) |
| Model comparison (exploratory) | every pair of models within an LLM condition | Same paired test |

"Deteriorating" means a vasopressor was started during the forecast horizon
(hours 24–48 after admission). In the lean run that was 110 of the 450 test
patients.

**Why a paired Wilcoxon test.** Per-patient errors are skewed (a few patients
are hard for every method) and far from normal. The same patients are scored
under every condition, so a paired, rank-based test fits and needs no
normality assumption.

**Why the bootstrap resamples patients.** The 24 × 5 cells of one patient are
strongly correlated. Resampling cells would treat them as independent and
give confidence intervals that are far too narrow. Resampling patients (2,000
resamples) respects the real unit of independence. The same reasoning is why
the test unit is the patient's mean error, not the individual cell.

---

## 2. The metrics

### 2.1 Point accuracy

**sMAPE (primary metric)**

`sMAPE = mean(2·|y − ŷ| / (|y| + |ŷ|))`, a fraction from 0 to 2.

- **Why:** it is unit-free, so heart rate (bpm), SpO2 (%) and lactate
  (mmol/L) can be pooled into one per-patient number for the tests. It is
  bounded, so one wild forecast can't dominate a mean. It is also the metric
  DT-GPT reports, the closest published LLM-for-clinical-forecasting work,
  which makes our results comparable.
- **Known weaknesses:**
  - It is not symmetric: under-forecasts are penalised more than
    over-forecasts.
  - It reaches its maximum of 2.0 whenever the forecast is 0 and the truth
    isn't. The Trillium run hit exactly this (section 3.2).
  - The per-patient pool is dominated by the dense vital signs. Lactate fills
    only about 10 % of cells, so it carries little weight. That's why per-variable
    numbers are always reported too.

**MAE and RMSE (secondary)**

- **Why:** they are in each variable's own unit, so they're clinically
  readable ("off by 13 bpm"). RMSE weights large errors more, so a large gap
  between RMSE and MAE flags outliers. That gap is what exposed the GBM's wild
  MAP forecasts on Trillium (RMSE 90.5 vs MAE 13.9).
- **Why not primary:** they can't be pooled across variables with different
  units.

### 2.2 Realism

**KS statistic (per variable)**

The two-sample Kolmogorov–Smirnov statistic between forecast values and true
values. It asks whether the twin produces values in realistic ranges and
proportions, not only close values. The statistic is the effect size; its
p-value is meaningless at these sample sizes.

Only cells with both a true value and a forecast are compared. Comparing all
forecasts with the few true lactate values compared lactate in general against
lactate drawn when patients are sick (section 3.2).

**Cross-variable correlation preservation**

The Frobenius norm of the difference between the true and forecast
variable-by-variable correlation matrices. A digital twin should keep
physiology's co-movement, for example heart rate and respiratory rate rising
together. No point-accuracy metric measures that.

### 2.3 Uncertainty

**Interval coverage and width (per variable)**

Each method gives a 90 % interval: the LLM states a half-width, and GBM and
LSTM derive one from their training residuals.

- **Coverage** is the share of true values inside the interval. The target is
  0.90: lower means overconfident, higher means wasteful.
- **Width** is the interval's mean width, in the variable's own unit, which is
  why it's reported per variable and never pooled. Coverage alone can be gamed
  with huge intervals, so the two are always read together.

**Split-conformal calibration** (`calibrate.py`)

On the calibration patients, the score of each observed cell is
`|y − ŷ| / halfwidth`. The calibration factor is that score's conformal
quantile at 90 %. Multiplying the test half-widths by the factor gives about
90 % coverage without any distributional assumption.

- **Why:** every method then reaches the same coverage, so methods can be
  compared fairly on width: after calibration, a narrower interval is a better
  interval.
- **A diagnostic in its own right:** the factor itself shows how
  trustworthy a method's own uncertainty is. 1 means honest, 10 means the
  stated interval was ten times too narrow.

### 2.4 Safety

**Plausibility-violation rate**

The share of forecast values outside a clinically plausible range per
variable (`plausible_range` in the config, e.g. heart rate 20–250). An
implausible value from a digital twin is a safety problem even when it
doesn't hurt the average much. This is RQ2's primary outcome.

The critic always ends by clipping to that range, so its post-critic rate is
0 by construction. The critic's real evidence is therefore:

- the out-of-range rate **before** it acts;
- the share it fixed with an LLM correction rather than clipping;
- the comparison with a clip-only critic (`full_pipeline_clip_critic`).

All three are recorded since the changes in section 3.3.

### 2.5 Pipeline health (not accuracy, but read first)

- **Fallback rate:** the share of LLM forecasts that failed (unparseable,
  wrong shape, time-out) and were replaced by the naive forecast.
- **Fill rate:** the share of variables the LLM left out and that were
  filled.

A model with many fallbacks just looks like the naive baseline, so these are
checked before any accuracy comparison. Tuning refuses settings with more
than 10 % fallbacks. The Trillium run had 0 % of both in every condition.

### 2.6 Metrics considered and not (yet) used

| Metric | What it would add | Why it isn't used (or not yet) |
|---|---|---|
| MAPE | the familiar percentage error | Undefined or explosive when the truth is near 0; sMAPE is the bounded version |
| MASE (error relative to the naive forecast) | Directly answers "better than persistence?", unit-free | A strong candidate. It wasn't in the original plan, and the paired comparison with the naive condition answers the same question. Worth adding as a secondary table |
| CRPS, log-likelihood | Proper scores for a whole predictive distribution | The LLM gives one point forecast and a symmetric half-width, not a distribution; scoring a distribution it never stated would mean inventing one |
| Interval (Winkler) score, pinball loss | One number that trades width against misses | Usable with what we have. Coverage plus calibrated width gives the same information more readably; a reasonable addition |
| Wasserstein distance, MMD | Distribution distance with magnitude (KS is shape only) | KS is standard and easy to read; these add little for one-dimensional marginals |
| Dynamic time warping, trajectory shape | Credit for the right shape at a slightly wrong time | Hourly ICU forecasts are judged hour by hour; DTW would forgive late warnings, which clinically shouldn't be forgiven |
| AUROC, alarm precision and recall | Clinical usefulness: does the twin predict deterioration? | That is a classification task with its own labels and thresholds; RQ3 instead asks whether forecast accuracy holds in deteriorating patients. A natural follow-up study |
| Clinician rating of plausibility | Face validity beyond fixed ranges | Out of scope (needs clinician time and ethics approval for review); fixed ranges are the reproducible proxy |

---

## 3. The first full lean run on Trillium: what we learned and changed

### 3.1 The results in one paragraph

All 19 conditions ran on the 450 test patients with no fallbacks.

- **RQ1:** supported (full pipeline 0.163 vs single model 0.180 per-patient
  sMAPE, p = 8e-23, and the same direction for every model).
- **RQ2:** the critic doesn't change accuracy (p = 0.067), and a clip-only
  critic does as well.
- **The important result:** no LLM condition beat the naive baseline (0.136),
  and GBM (0.122) beat every LLM condition. The best LLM setup
  (Med42-70B, full pipeline, 0.136) only tied naive.
- **Intervals:** the LLMs' 90 % intervals covered only 16–39 % of outcomes.

The question was why. The rest of this section is what the diagnosis found.
All of it comes from the saved forecasts plus the observation windows rebuilt
with the project's own code.

### 3.2 What we learned

**1. The LLMs extend the given trend in a straight line.**

The prompt gave each variable's `trend_slope`. We regressed each forecast's
24 h change on `23 × trend_slope`; a coefficient of 1 means the model drew a
straight line from the slope.

| | HR | RR | SpO2 | MAP |
|---|---|---|---|---|
| truth | −0.12 | −0.18 | −0.09 | −0.17 |
| GBM | −0.14 | −0.16 | −0.11 | −0.17 |
| single-model LLMs (6 models) | 0.95 – 1.17 | 0.86 – 1.15 | 0.97 – 1.11 | 0.96 – 1.53 |
| full pipeline, Med42-70B | −0.13 | −0.15 | −0.06 | −0.12 |

Patients revert toward their mean; the LLMs continue the trend. That is why
LLM error grew steeply with lead time (hour 1 ≈ 0.09 for everyone; hour 24
0.25–0.31 for single models against 0.14 for naive). It also explains most of
RQ1's effect: the similar patients' trajectories pull the model away from the
straight line, and the models that use them best rank highest.

**2. An unobserved variable became a forecast of 0.**

36 % of test patients had no lactate in their first 24 h. The prompt said
only "no observations", and models without the similarity context then wrote
0 in 18–37 % of lactate cells. That gives the maximum sMAPE (2.0) and counts
as implausible. The naive baseline had the same flaw in code (`0.0` as the
"last value").

**3. Charting artefacts reached every model.**

The extracted panel contained values such as MAP 79,104 and −5, SpO2 10,099,
heart rate 0, and respiratory rate 0 and 222. Nothing filtered them.

- The GBM learned from artefact targets and forecast MAP between −2,759 and
  5,395 for 11 % of test patients.
- The artefacts also inflated the LSTM's interval widths and skewed the
  similarity features and the prompts' statistics.

**4. The LLMs' stated uncertainty is uninformative.** Calibration had to
inflate LLM intervals 3–16×, against about 1–2× for the baselines. Even after
calibration they were wider than GBM's. The model gave one fixed half-width
per variable, the same at hour 1 and hour 24.

**5. Two metrics were misleading.**

- KS compared all forecasts with sparse true lactate values, giving about 0.35
  for every method whatever its quality.
- A pooled "mean interval width" averaged bpm, % and mmol/L.

**6. The critic's headline was guaranteed by design.** Its 0 % violation rate
follows from its final clipping step, so it can't count as evidence.

### 3.3 What we changed

Corrections (bugs or flaws, applied everywhere):

| # | Change | Fixes |
|---|---|---|
| C1 | Raw values outside a per-variable `valid_range` (default `plausible_range`) are dropped at extraction and again when the hourly arrays are built | artefacts (3) |
| C2 | A variable with no observations is filled with the training median (± 1.645 SD), not 0, in the naive baseline and the LLM fallback | zeros (2) |
| C3 | The prompt names an unobserved variable's typical cohort value ("not measured; median 2.1 mmol/L, IQR 1.4–3.2") | zeros (2) |
| C4 | KS compares only cells with both a true value and a forecast | misleading KS (5) |
| C5 | GBM and LSTM outputs are clipped to the valid range. Coverage and width are reported per variable. The critic now records the out-of-range values it saw and how many its LLM fixed vs clipped | artefacts (3), misleading width (5), critic evidence (6) |

Improvements, as new `forecasting_agent` options. Each defaults to the old
behaviour, so the first run's settings stay reproducible:

| Option | Values | Aimed at |
|---|---|---|
| `trend_hint` | `slope` (old), `none` (no slope in the prompt), `damped` (slope plus a note that ICU values revert to the mean) | trend extrapolation (1) |
| `drift_damping` | λ in [0, 1]: each hour's change from hour 1 is multiplied by λ (1 = old, 0 = flat) | trend extrapolation (1) |
| `cohort_anchor` | adds the training cohort's median [IQR] of every variable to the prompt | trend (1) and zeros (2) for the single model, which has no similar patients to anchor on |
| `interval` | `constant` (old), `endpoints` (half-width at hour 1 and hour 24, interpolated; the new default in the Alliance configs), `per_hour` (all 24) | flat, uninformative intervals (4) |

Three new tuning rounds (`trend_hint`, `drift_damping`, `cohort_anchor`) let
`tune.py` choose among these on the tuning patients.

### 3.4 How we knew the changes might help, and the limits of that evidence

**The corrections** follow directly from the data. A MAP of 79,104 is not a
measurement, and a forecast of 0 mmol/L lactate is not a forecast. Removing
them can only make the evaluation more faithful. We also expect them to help
GBM most, because its targets were the most contaminated, so the gap between
the LLMs and GBM may well **widen** in the second run.

**The improvements** were motivated by a counterfactual on the first run's
own forecasts: simple post-processing changes applied to the saved
forecasts, scored against the same truth.

| Condition | As run | Change from hour 1 × 0.5 | Held flat at hour 1 | Average with GBM |
|---|---|---|---|---|
| single model (Qwen) | 0.180 | 0.154 | 0.136 | 0.135 |
| full pipeline (Qwen) | 0.163 | 0.143 | 0.132 | 0.129 |
| full pipeline (Med42-70B) | 0.136 | 0.124 | 0.123 | 0.123 |
| *naive 0.136, GBM 0.122* | | | | |

Shrinking the extrapolated trend improved every LLM condition, and holding
each LLM's own first-hour value flat took the primary pipeline from worse
than naive to better than it. Averaging with GBM never beat GBM alone. That
says the LLMs, as prompted, carry no information GBM doesn't already have,
which is a result in itself.

**This evidence has three limits:**

1. **It used the test set.** Picking a damping value or prompt from this table
   would be tuning on the test patients. That is why the changes are options
   chosen by `tune.py` on validation patients, not hard-coded winners.
   Still, the *idea* of trying them came from looking at test results. For
   the thesis this should be reported plainly, and the second run's results
   are best read as a confirmation of a diagnosed failure mode, not as a
   fresh, untouched test. A stricter design would draw a new test split, or
   hold out a later admission period, for the final claim.
2. **Post-processing is not the same as prompting.** `drift_damping` is the
   post-processing tested above. `trend_hint` and `cohort_anchor` change what
   the model reads, and whether models follow such instructions can only be
   measured by running them. The tuning rounds do that.
3. **The intervals have no counterfactual.** A model can't be re-asked
   afterwards for a widening interval. The expectation rests on the observed
   failure (one fixed width for hours 1 and 24 while errors grow 2–3× over
   the horizon). The test is the second run's calibration factors: factors
   closer to 1 and narrower calibrated widths than in the first run would
   mean the stated uncertainty has become informative.

### 3.5 What the second run (`tri_lean_exp2`) should be judged on

- **Primary:** is any LLM condition significantly better than naive (median
  fill), and how far is the best one from GBM?
- **Trend coefficients** (section 3.2, item 1) should move from about 1
  toward the truth's −0.1 to −0.2 for the single-model conditions.
- **Lactate:** no more forecasts of 0; sMAPE on unobserved-lactate patients
  well below 2.0.
- **RQ2:** the before-critic violation rate and the LLM-fixed share, next to
  the clip-only comparison.
- **Intervals:** calibration factors closer to 1, and calibrated per-variable
  widths compared with GBM's.
- **Tuning:** which `trend_hint` / `drift_damping` / `cohort_anchor` won on
  validation, and by how much. If `flat` (λ = 0) wins, the LLM is contributing
  a level estimate only, which is worth saying in the thesis.

---

## 4. Hyperparameters: what was tuned, how, and what changed

### 4.1 How they are optimized

`src/tune.py` with `config/tuning_grid.yaml` (run by the `tune` stage):

- **Data:** the 128 tuning patients of the validation split. Never the test
  split, and never the 322 calibration patients.
- **Objective:** mean per-patient sMAPE, averaged over the round's conditions.
  A setting is **eligible** only if every condition's LLM fallback rate is at
  most 10 %, so a setting can't win by failing over to the naive forecast.
- **Search:** coordinate search. The rounds run in order. Each round
  evaluates the current best settings (the *incumbent*) and every trial in
  it, the incumbent plus that trial's overrides. A trial replaces the
  incumbent only if it lowers the objective by at least **0.002** (0.2
  sMAPE points). Smaller gains are treated as noise at n = 128.
- **Fixed for every trial:** the LSTM seed (42), so LSTM trials differ by
  their settings, not by their random initial weights.
- **Caching:** every result is cached by its settings, so an incumbent already
  scored in an earlier round costs nothing.
- **Output:** the winning overrides are written into the tuned config
  (`<config>_tuned.yaml`), which calibrate, run and evaluate use.
- **Which models:** tuned with the primary LLM (Qwen2.5-32B) only, and the
  winners apply to every model variant.

**Why coordinate search, not a full grid or Bayesian optimisation.** One LLM
evaluation on 128 patients costs 10–20 GPU-minutes. A full grid over these
options would be hundreds of GPU-hours. Coordinate search scores each option
once against the current best, which is cheap enough to run before every
experiment. The rounds are ordered so the settings that most change what the
model sees come first. It can miss interactions between settings in
different rounds. That's an accepted limitation.

### 4.2 What was searched in the Trillium run, and what won

Starting values come from the base config (`config_alliance_lean.yaml`) or
the code defaults. The scores are mean per-patient sMAPE on the 128 tuning
patients (`results/tuning/trials.csv`).

| Round | Conditions scored | Starting value | Trials (objective) | Result |
|---|---|---|---|---|
| gbm | gbm | depth 4, 150 iterations, learning rate 0.1 (0.1179) | depth 3 (0.1167), depth 6 (0.1200), 300 iterations at 0.05 (0.1173) | kept: depth 3 gained only 0.0012 |
| lstm | lstm | hidden 128, 200 epochs, learning rate 1e-3 (0.1289) | **hidden 64 (0.1184)**, hidden 256 (0.1210), 400 epochs (0.1332), learning rate 3e-3 (0.1259) | **hidden 64** |
| similarity_context | full_pipeline | `horizon_mean`: one neighbour average per variable (0.1653, with 6.25 % fallbacks) | **`trajectory`**: neighbours' median and IQR at 5 horizon hours (0.1629) | **trajectory** |
| k_neighbors | full_pipeline | 30 (0.1629) | 10 (0.1645), **50 (0.1602)** | **k = 50** |
| prompt | single_model_llm + full_pipeline | as before (0.1677) | `strict_length` (0.1669), last 6 hourly values (0.1705), both (0.1696) | kept |
| temperature | single_model_llm + full_pipeline | 0.2 (0.1677) | 0.0 (0.1672) | kept |
| critic | full_pipeline | 1 correction attempt (0.1602) | 2 attempts (0.1618) | kept |

The tuned config changed three settings: LSTM hidden size 64, similarity
context `trajectory`, and k = 50. Two notes:

- The `horizon_mean` score came from an early tune job in which 6.25 % of
  forecasts failed, before schema-constrained decoding fixed wrong-length
  outputs. So that one comparison isn't fully clean. The failures fell back
  to naive, which scored better than the LLM, so if anything they favoured
  the setting that lost.
- These are validation scores. The test run doesn't re-score the losing
  settings, so it can't confirm the choices, only show that the tuned
  pipeline's similarity component helps (no-similarity ablation 0.176 vs
  0.163 on the test patients).

**Not tuned, on purpose:**
- **Fixed by the study design:** the observation and forecast windows
  (24 h / 24 h), the variable panel and the cohort criteria. Changing them
  changes the question.
- **Compared, not tuned:** the choice of LLM. Each model is its own
  condition.
- **Operational, not methodological:** `num_ctx`, `max_tokens`, request
  time-outs and concurrency. They are set so that nothing is truncated and
  checked through the fallback rate.
- **Not judged by the objective:** interval settings. Tuning scores accuracy,
  and intervals are calibrated separately (section 2.3).

### 4.3 What changed after the Trillium run, and why

**Three rounds were added**, placed after `k_neighbors` and before `prompt`:

| Round | Trials | Why |
|---|---|---|
| `trend_hint` | `none` (no slope in the prompt), `damped` (slope plus a mean-reversion note) | The models extended `trend_slope` about 1:1 for 24 h (section 3.2, item 1) |
| `drift_damping` | λ = 0.5, 0.25, 0 (flat at the model's own hour 1) | The counterfactual in section 3.4: shrinking the drift improved every LLM condition. Tuning now picks λ on validation patients instead of the test patients that suggested it |
| `cohort_anchor` | on | The single model has no similar patients to anchor on and produced zeros and runaway trends; the cohort's median and IQR give it a reference point |

All three score both `single_model_llm` and `full_pipeline`. These options
change the forecaster used by both RQ1 arms, and a setting that helps one arm
and hurts the other must not win on the strength of one.

The earlier prompt rounds stay. Their winners may change now that the
trend and anchor settings come first.

**Interval mode is set, not tuned.** The objective is sMAPE, which can't tell
a good interval from a bad one, so `interval` is a design choice:
`endpoints` in the Alliance configs (section 3.3). Its effect is judged by
the calibration factors in the second run.

**Every round runs again from scratch.** The data changed (artefact filter
and median fill, section 3.3), so all cached trials are invalid. The
checkpoints carry a data-version marker. Each round therefore starts again
from the base-config values, and last time's winners (hidden 64, trajectory,
k = 50) have to win again on the cleaned data. That is deliberate: settings
chosen on contaminated data shouldn't be carried over unchecked.

**Unchanged:**
- The acceptance rule (0.002 gain, at most 10 % fallbacks) and the 128-patient
  tuning subset. Changing them after seeing results would move the goalposts.
- The GBM, LSTM, similarity and critic grids. Depth 3 GBM narrowly missed the
  threshold and is tried again on the cleaned data.

**Cost:** the grid grows from 14 to 27 LLM evaluations, about 7 GPU-hours
instead of 3.5. `jobs/run_all.sh` estimates the tune stage at 400 minutes for
lean.

**Possible next additions** (not in the grid yet):
- k = 100: 50 was the largest value tried, and it won.
- A trial combining `trend_hint: damped` with `drift_damping`, in case the
  prompt change alone is not enough.
