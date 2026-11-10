# Evaluation — how this project judges an experiment

This page explains how the experiment's results are evaluated, why each
metric was chosen (and others weren't), what the first full lean run on
Trillium taught us and what we changed because of it, how the
hyperparameters are tuned (section 4), what the second run
(`tri_lean_exp2`) showed and changed (section 5), and what the third
(`tri_lean_exp3.4`) showed and changed (section 6), and what the full-scope
tuning on Fir (`fir_full_v2`) changed (section 7). Every number comes from that run, or from re-analysing
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
- **Stronger persistence (added after the second run):** `recent_mean` (the
  mean of the last 6 observed hours, held flat) and `persistence_blend` (half
  the last value + half the 24 h mean). An LLM that beats only the naive
  baseline may just be estimating a better level.
- **LLM conditions:** `single_model_llm` (the LLM alone, DT-GPT style),
  `full_pipeline` (similarity agent + forecaster + critic), and three
  ablations of it: `_no_critic`, `_no_similarity` and `_clip_critic`. Each
  runs with the primary model and, as `@variant`, with the other local
  models. The primary model was Qwen2.5-32B in the first two runs and
  Med42-70B since then (section 5); Qwen stays in the comparison as the
  `qwen2_5_32b` variant.
- **Ensembles (added after the second run):** e.g. `full_pipeline+lstm`, the
  mean of its members' forecasts, built from the saved forecasts at no GPU
  cost and compared with each member.

### 1.4 The steps

1. **Tune** (`tune.py`, on the 128 tuning patients). A coordinate search: each
   round tries a few settings and keeps one only if it lowers the mean
   per-patient sMAPE by at least 0.002, with at most 10 % of LLM forecasts
   falling back to naive.
2. **Calibrate** (`calibrate.py`, on the 322 calibration patients). Since
   the third run it first fits each LLM condition's post-processing (drift
   damping and level anchor, section 5.3) from its saved forecasts. Then,
   for each condition and variable, a split-conformal factor scales the
   forecast's interval so it covers about 90 % of outcomes (section 2.3).
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


---

## 5. The second lean run (`tri_lean_exp2`): what we learned and changed

The run's full write-up is in its results folder (`README.md` next to
`mimic-iv-twin-work/`). Its test patients are almost all new: the artefact
filter changed which stays were eligible, and only 6 of the 450 overlap with
the first run. (The sample itself also wasn't reproducible until section
7.2's fix, so a new extraction drew new patients anyway.) That largely answers limitation 1 of section 3.4. The changes
were diagnosed on the first run's test patients and confirmed here on new
ones.

### 5.1 What the section 3.5 checks showed

| Check | Result |
|---|---|
| Any LLM condition better than naive? | Yes. Best: Med42 and Llama-3-70B pipelines, 0.111 vs 0.123 (p ≈ 1e-22). Every LLM pipeline beat naive |
| Distance to the best classic model | Level with GBM (0.1115, p = 0.07); 0.001 behind LSTM (0.1094, p = 0.06), now the best method |
| Trend coefficients | Single models moved from 0.95–1.5 to between −0.11 and +0.09 (truth −0.06 to −0.15) |
| Lactate | No forecasts of 0; sMAPE on unobserved-lactate patients about 0.4 instead of 2.0 |
| RQ2 | Before the critic, 0.05 % of values were out of range, and its LLM fixed all of them; no accuracy effect (p = 0.37). Clipping does as well |
| Intervals | They now widen 1.7–2× over the horizon, but conformal factors are still 3–11×; after calibration, LLM widths match LSTM's |
| Tuning | `trend_hint: damped` and `drift_damping: 0.25` won (objective 0.160 → 0.126); damping values 0.5 to 0 all scored within 0.002 |

### 5.2 What we learned

1. **The LLMs mostly estimate a level.** With damping at 0.25 the forecast is
   nearly flat. The best pipelines tie simple persistence: half the last value
   plus half the window mean scores 0.1107; the 6 h mean, 0.1127.
2. **They anchor on the noisiest level.** The LLM's hour-1 value is the last
   reading (mean difference about 0.01). A 6 h mean is a better level than
   the last value (0.1127 vs 0.1229).
3. **They carry information the classic models don't.** Averaging Med42's
   pipeline with LSTM gives 0.1075, better than LSTM alone (p = 0.0008) and
   better than GBM + LSTM (0.1082). In the first run, averaging with a
   baseline never helped.
4. **RQ1 depends on the model.** For Qwen, the primary model, the full
   pipeline is now slightly *worse* than the single model (0.1214 vs 0.1199,
   p = 0.025): damping does the job the similarity context used to do. For
   the other five models the pipeline still wins (by 0.003–0.021).
5. **Two output faults.** Single-model Med42 produced values like
   −3 × 10⁹ for 4 patients, and single models forecast SpO2 above 100 % in
   70–631 cells. sMAPE hides both, but they corrupt MAE, RMSE and bias, and
   they are impossible values for a digital twin.

### 5.3 What we changed

| Change | From finding |
|---|---|
| **Output sanitising:** a variable with non-finite values, or values outside 10× its valid range, is filled like a left-out variable; every LLM forecast is clipped to the variable's `hard_range` (SpO2 [0, 100]; others ≥ 0). This is a physical limit, not a plausibility check, so the critic comparison is unaffected | 5 |
| **Level anchor:** post-processing moves hour 1 a share `level_anchor_weight` of the way to the mean of the last 6 observed hours, before damping | 1, 2 |
| **Post-processing fitted per condition:** `calibrate.py` chooses `drift_damping` (0.1–1) and `level_anchor_weight` (0–0.75) for each LLM condition on the calibration patients, from the forecasts saved before post-processing (no LLM calls), and `run_experiment.py` applies them on the test patients | 1, 2, and the damping plateau in tuning |
| **Persistence baselines** `recent_mean`, `persistence_blend` | 1 |
| **Ensemble** `full_pipeline+lstm` | 3 |
| **Primary model Med42-70B**, Qwen kept as a variant | 4 |

### 5.4 Hyperparameter changes, and why

- **The `drift_damping` tuning round was removed.** Damping is
  post-processing, so it doesn't change what the LLM is asked. Re-running the
  LLM for each value (6 evaluations, about 1.5 GPU-hours with Med42) is
  wasted. Fitting it per condition from saved forecasts is free and lets
  each model have its own value.
- **The fitting grid is bounded.** Damping is at least 0.1 and the anchor
  weight at most 0.75. Without the bounds the fit can choose "anchor 1,
  damping 0", which *replaces* the LLM by the `recent_mean` baseline. The
  end-to-end test with a mock LLM did exactly that. The condition must keep
  measuring the LLM.
- **Tuning is now done with Med42**, the new primary model, so the prompt
  settings are chosen for the model the RQs are reported on.
- **The tuning starting point is `drift_damping: 0.25`** (the last tuned
  value) in the base configs. Later tuning rounds are then scored on damped
  forecasts, as the final ones will be.
- **Unchanged:** the acceptance rule, the 128-patient tuning subset, and the
  other rounds.

**Cost of fitting on the calibration patients.** The same 322 patients now
fit two post-processing parameters per condition and then the conformal
factors. Two parameters on 322 patients carry little risk of overfitting, but
the conformal coverage guarantee strictly needs patients the parameters
weren't fitted on. Read coverage within about ±0.02 of 0.90 as calibrated.

### 5.5 What the third run should be judged on

- **Primary (Med42) full pipeline:** about 0.110 or better, and its
  comparison with `recent_mean` and `persistence_blend`. Beating them is the
  claim that the LLM adds more than a level.
- **The fitted parameters:** an anchor weight above 0 for most LLM
  conditions confirms finding 2; damping near 1 for a model would mean its
  trends have become useful.
- **The ensemble** `full_pipeline+lstm` against LSTM.
- **RQ1 with Med42** (single model 0.131 vs pipeline 0.111 in this run).
- **No impossible values:** all MAE/RMSE finite and plausible; SpO2 ≤ 100.

---

## 6. The third lean run (`tri_lean_exp3.4`): what we learned and changed

Its own extraction (450 test patients; before section 7.2's fix, so not the
same patients as `tri_lean_exp2`), Med42-70B as the primary model, post-processing fitted per condition. Per-patient sMAPE,
paired Wilcoxon tests.

### 6.1 What the section 5.5 checks showed

| Check | Result |
|---|---|
| Med42 full pipeline ≈ 0.110 or better | 0.1095 |
| vs `recent_mean` / `persistence_blend` | 0.1142 (p ≈ 1e-10) / 0.1128 (p ≈ 1e-5): the LLM adds more than a level |
| vs the classic models | LSTM 0.1113 (p = 0.03), GBM 0.1143 (p ≈ 2e-14): the first LLM condition ahead of LSTM. These two tests were computed outside `evaluate_results.py` |
| Fitted parameters | Anchor weight > 0 everywhere (0.5 for every pipeline, 0.75, the bound, for 5 of 6 single models); damping 0.25 or 0.1, never near 1 |
| Ensemble | 0.1092: better than LSTM (p ≈ 5e-9), not than the pipeline alone (p = 0.11) |
| RQ1 | Pipeline 0.1095 vs single model 0.1118 (p ≈ 4e-5); without similarity 0.1119. The gap shrank from 0.02 because the single model gets the level anchor too |
| Impossible values | None: no fallbacks, fills, non-finite values or SpO2 > 100 |
| Coverage after calibration | 0.881–0.893 on the test patients |

Pairs of conditions that are the same setup (the critic never acted, see
6.2) score within 0.0003–0.0005 of each other: the noise of re-running the
LLM. The 0.0018 gap to LSTM is several times that.

### 6.2 What we learned

1. **Zero-width intervals.** The LLMs sometimes give a half-width of 0,
   mostly for lactate (17 % of single-model Med42 forecasts; Baichuan for
   every variable in 8 patients). Conformal factors scale the half-width, so
   these never widen; `calibrate.py` leaves them out of the fit, and they
   almost never cover. `full_pipeline@baichuan_m2` reached only 0.864 on the
   calibration patients themselves; lactate coverage was the lowest
   (0.81–0.88).
2. **RQ2 measured nothing.** The critic reviewed the post-processed
   forecast, which the level anchor and damping keep close to the observed
   values: 0 implausible values in every condition, so the critic,
   no-critic and clip-only conditions were the same setup re-run.
3. **The tuning noise is as large as the tuning gains.** The same settings
   scored 0.1144 and 0.1130 in two tuning rounds; `cohort_anchor` (−0.0012)
   and `critic.attempts2` (−0.0013) lost within that noise, so the 0.002
   acceptance margin was right to reject them.
4. **The headline rests on one split.** p = 0.03 against LSTM on one set of
   450 patients.

### 6.3 What we changed

| Change | From finding |
|---|---|
| A half-width ≤ 0 is replaced: by the variable's smallest positive half-width if some hours have one, else by the naive interval (`ForecastingAgent._fix_zero_widths`, counted as `zero_interval`) | 1 |
| `critic_agent.stage: raw` (default `final`): the critic reviews the LLM's own output, then post-processing is applied to the reviewed forecast | 2 |
| `run_experiment.py` saves the forecast before post-processing (`y_raw`) with each LLM condition; `evaluate_results.py` reports its violations (`<condition>_before_postprocess`). `src/raw_violations.py` counts them from an older run's checkpoints | 2 |
| `[full_pipeline, lstm]` and `[full_pipeline, gbm]` in `extra_comparisons` | 6.1 |
| `jobs/run_all.sh --config <file>` (another base config for a run), `run_all.estimates_min` and `tuning_grid` in a config | 3, 4 |

### 6.4 The next runs and what they should be judged on

- **`tri_lean_exp4`** (`config/config_alliance_lean_exp4.yaml`): the same
  experiment on a new 3000-stay sample (`random_seed: 43`), tuned again,
  with only the conditions the claims need. Judged on whether the pipeline
  again beats LSTM, GBM and the persistence baselines, and RQ1 holds;
  coverage within ±0.02 of 0.90, lactate included, with `zero_interval`
  counted.
- **RQ2 run** (`config/config_alliance_lean_rq2.yaml`): exp3.4's cohort and
  tuned settings with the critic before post-processing, on Llama-3-70B:
  `full_pipeline`, `full_pipeline_no_critic`, `full_pipeline_clip_critic`.
  `src/raw_violations.py` on exp3.4 decided the model: before
  post-processing, Med42's forecasts were all plausible (0 of 450 test
  patients), while the other models had 0.01–0.56 % of values out of range,
  every one *below* the range, mostly lactate under 0.1 mmol/L and
  respiratory rate under 4. Llama-3-70B, Med42's base model, had the most
  in its pipeline (53 of 450 patients) and tied Med42 for accuracy.
  Post-processing removed every one of them in every condition. Judged on
  the share its LLM critic fixes rather than clips, and sMAPE against no
  critic and against clipping; with 0.56 % of values affected, an sMAPE
  difference is unlikely to exceed the 0.0003–0.0005 re-run noise.

---

## 7. Full-scope tuning on Fir (`fir_full_v2`): what we learned and changed

19 variables (14 of them labs), Med42-70B, stopped after tuning
(`--until tune`). No fallbacks or filled variables in any trial; the longest
request was ~6,000 of 8,192 tokens, none truncated.

### 7.1 What we learned

1. **One shared prompt setting can favour one RQ1 arm.** `trend_hint:
   none` took the single model from 0.1233 to 0.1100 but the pipeline from
   0.1121 to 0.1185 (about 4x the tuning noise), and won on the mean of the
   two. At the lean scope the same choice cost the pipeline nothing
   (0.1139 -> 0.1130).
2. **The LSTM was handicapped at this scope.** 0.140 against GBM's 0.109
   (lean: LSTM 0.110, the best baseline). It saw unmeasured hours as the
   cohort mean, so a lab measured every 6-24 h read as the cohort mean most
   hours; GBM gets the last measured value from its summary features.
3. **Ollama can silently fall back from CUDA to Vulkan.** Its GPU discovery
   gives each CUDA library 30 s to load; from `$HOME` on Fir that timed out,
   and job 63850581 ran on Vulkan, 1.65x slower (1,420 vs ~850 s per batch).
4. **Runtime as estimated.** ~850 s per batch of 32 for a Med42 condition
   on CUDA, 3.7x the lean time per patient, as `run_all.sh` assumes.

### 7.2 What we changed

| Change | From finding |
|---|---|
| `forecasting_agent.per_condition` (`common.py`): prompt settings for one group, `single_model_llm` or `full_pipeline` (every `full_pipeline*` condition), over the shared ones. Rounds with `per_condition: true` in `tune.py` pick a winner per group; `trend_hint`, `cohort_anchor` and `prompt` are such rounds now. Checkpoints are keyed by each group's effective settings, so trials already run are reused | 1 |
| `baselines.lstm_input: ffill_mask` (`baselines.py`): the last measured value carried forward plus a measured/not channel per variable; set in the lean, full and exp4 configs (the default, `zero_fill`, keeps older runs reproducible) | 2 |
| `jobs/ollama_lib.sh`: the libraries are read into the page cache before `ollama serve`, `OLLAMA_VULKAN=false`, and a job stops unless Ollama found the GPU through CUDA (`DT_OLLAMA_REQUIRE_CUDA=0` skips the check) | 3 |
| `extract_cohort.py`: eligible stays sorted by `stay_id` before the seeded 3000-stay sample. It sampled by row position from a DuckDB result without `ORDER BY`, so the same seed drew a different cohort per extraction: the Nibi RQ2 run shared 1509 of 3000 stays (214 of 450 test patients) with exp3.4, from identical raw panels. That run uses exp3.4's `cohort.parquet` and `panel_long.parquet` instead; runs extracted before the fix keep their own files as the record of their cohort | — |

Replaying `fir_full_v2`'s tuning with these rules: the single model keeps
`trend_hint: none`, the pipeline the default (`slope`; `damped_note` gained
0.00198, just under `min_improvement`). The re-run needs 6 new Med42
pipeline evaluations and the LSTM trials.

---

## 8. Pooling the three lean samples (`src/pool_runs.py`)

`tri_lean_exp3.4` (seed 42), `tri_lean_exp4` (43) and `ror_lean_exp5` (44)
each test the claims on a different sample of 450 test patients (17 test
patients appear in two samples). `src/pool_runs.py` reads their saved
forecasts and repeats each paired comparison per sample, pooled over all
1,350 patients (with a patient-bootstrap CI), and within the RQ3 subgroups,
with a Mann-Whitney test of whether the difference changes between them. No
GPU: it runs in seconds on a login node. The samples and comparisons are in
`config/pooled_lean.yaml`, whose header says how to lay out the packed
runs.

exp3.4's LSTM saw unmeasured hours as the cohort mean (`zero_fill`); exp4 and
exp5 use `ffill_mask` (7.2). `tri_lean_exp3.4_lstm`
(`config/config_alliance_lean_exp3.4_lstm.yaml`) re-trained the `ffill_mask`
LSTM on exp3.4's patients (its naive forecasts are identical to exp3.4's,
so the split is the same), and the spec takes sample s42's LSTM from it.

Results (per-patient sMAPE, A - B; Wilcoxon p):

| Comparison | s42 | s43 | s44 | Pooled (n = 1,350) |
|---|---|---|---|---|
| RQ1: pipeline - single model | -0.0023 (4e-5) | -0.0030 (0.001) | -0.0033 (4e-8) | -0.0029 (3e-13) |
| pipeline - LSTM | -0.0010 (0.053) | -0.0016 (0.16) | -0.0015 (0.17) | -0.0014 (0.007) |
| pipeline+LSTM - LSTM | -0.0019 (2e-8) | -0.0026 (7e-11) | -0.0026 (2e-9) | -0.0024 (1e-25) |
| pipeline+LSTM - pipeline | -0.0008 (0.035) | -0.0011 (3e-4) | -0.0011 (8e-4) | -0.0010 (1e-7) |
| pipeline - GBM | -0.0048 | -0.0045 | -0.0054 | -0.0049 (3e-34) |
| pipeline - recent_mean | -0.0047 | -0.0037 | -0.0049 | -0.0044 (6e-29) |
| pipeline - no similarity | -0.0024 | -0.0045 | -0.0053 | -0.0041 (8e-27) |
| Med42 - Llama-3-70B pipeline | -0.0001 (0.35) | -0.0008 (0.064) | -0.0012 (6e-5) | -0.0007 (1e-4) |

RQ3 subgroups (pooled, exploratory: noticed in exp4 before the pooling was
specified): the pipeline beats the single model alike in stable (-0.0028)
and deteriorating patients (-0.0032; interaction p = 0.94). Its lead over
the LSTM is mostly in deteriorating patients (-0.0032, p = 0.01, vs stable
-0.0006, p = 0.12; interaction p = 0.11; seen in s43 and s44, not s42), and
so is the ensemble's (-0.0037 vs -0.0019; interaction p = 0.013).


### 8.1 Does the comparison with the LSTM depend on its seed?

Every run trained its LSTM with seed 42 only. `src/lstm_seeds.py` re-trained
it with seeds 42-46 on each sample's patients (Trillium job 1078102,
Rorqual job 22994509; re-training seed 42 reproduced each run's LSTM exactly),
and `config/pooled_lean_lstm_seeds.yaml` repeats the comparisons with each.
Test sMAPE of the LSTM varies by up to 0.0022 between seeds within a sample
(s42 0.1099-0.1117, s43 0.1089-0.1111, s44 0.1089-0.1097).

| Pooled over 1,350 patients | seed 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| pipeline - LSTM | -0.0014 (p = 0.007) | -0.0018 (7e-4) | -0.0018 (9e-5) | -0.0019 (2e-4) | -0.0014 (0.003) |
| ensemble - LSTM | -0.0024 (1e-25) | -0.0027 (4e-29) | -0.0026 (2e-31) | -0.0026 (4e-28) | -0.0024 (5e-26) |
| ensemble - pipeline | -0.0010 (1e-7) | -0.0009 (2e-6) | -0.0008 (2e-5) | -0.0006 (4e-5) | -0.0010 (1e-7) |
| pipeline - LSTM, deteriorating | -0.0032 (0.01) | -0.0039 (0.013) | -0.0034 (0.001) | -0.0046 (6e-4) | -0.0030 (0.023) |

Every conclusion holds for every seed, and seed 42, the one the runs used,
gives the pipeline its smallest lead over the LSTM (tied with 46): the
reported comparison is the conservative one. With the other seeds the
pipeline also beats the LSTM among stable patients (p = 0.012-0.035), so
"the lead is concentrated in deteriorating patients" is weaker than with
seed 42 alone (interaction p = 0.024-0.28).

### 8.2 Post-processing fitted separately for vital signs and labs

Fitted post-processing (5.3) uses one drift damping and one anchor weight
for all variables. `src/rescore_postprocess.py` refits it from the saved
pre-post-processing forecasts (`y_raw`), with no LLM calls, and writes the
result to `results_tuned/rescored_<strategy>_<grid>/`, leaving the run's own
results untouched. Its default (`--strategy global --grid standard`)
reproduces `calibrate.py`'s fit exactly: same settings, test sMAPE and
coverage, checked on tri_lean_exp3.4, tri_lean_exp4, ror_lean_exp5 and
ror_lean_fullcohort. On Fir's full-scope calibration forecasts, a two-fold
cross-validation on the calibration patients gave, against the current fit:
an extended grid (damping 0-1, anchor 0-1) -0.0003 on average; one setting
for the vital signs and one for the labs (`vitals_labs`) -0.0014 (better in
14 of 16 LLM conditions); one per variable -0.0012.

`vitals_labs` with the extended grid, applied to the three lean samples
(`config/pooled_lean_rescored.yaml`, pooled over 1,350 test patients):

| Comparison (A - B) | Runs' own fit | vitals_labs | 
|---|---|---|
| RQ1: pipeline - single model | -0.0029 (p = 3e-13) | -0.0020 (p = 2e-7); every sample p <= 0.006 |
| pipeline - no similarity | -0.0041 | -0.0029 (p = 1e-16) |
| pipeline - LSTM | -0.0014 (0.007) | -0.0015 (0.004) |
| ensemble - LSTM | -0.0024 (1e-25) | -0.0024 (1e-26) |
| ensemble - pipeline | -0.0010 (1e-7) | -0.0010 (2e-7) |
| pipeline - GBM / recent_mean | -0.0049 / -0.0044 | -0.0050 / -0.0045 |

Coverage after calibration is unchanged. The refit helps the single
models (0 to -0.0016 test sMAPE) and the pipeline without the
similarity agent (-0.0011 to -0.0013), not the full pipelines (-0.0003 to
+0.0003): the shared setting suited the pipelines and penalised the single
models, mostly on lactate. About a third of the RQ1 gap came from that, the
same kind of shared-setting bias as 7.1; RQ1 still holds on every sample,
but among deteriorating patients it is no longer significant (-0.0013,
p = 0.12). Whether to adopt the refit (for the lean and full scopes alike)
is open; it needs no GPU time, and the full scope's must wait for Fir.

### 8.3 Lactate intervals that under-cover

In ror_lean_fullcohort the calibrated lactate intervals of naive, the
persistence baselines and the single model covered 0.81-0.83 of test values
(target 0.90); tri_lean_exp3.4 showed 0.81-0.88. `src/diagnose_coverage.py`
(reproduces the calibration-set coverage, 0.900-0.904):

- It is a shift between calibration and test patients, not chance: the
  test nonconformity scores' 90th percentile is about twice the calibration
  one (KS p <= 2e-6), and 0.90 lies outside the patient-bootstrap range of
  the test coverage. GBM, the LSTM and (less so) the pipeline are not
  affected (0.88-0.92).
- The lactate factor rests on few patients (190 calibration patients with a
  lactate value in the full cohort, 110 in ror_lean_exp5) and heavy-tailed
  errors, so it is unstable: in ror_lean_exp5 naive's factor is 6.3 with a
  bootstrap range of 3.3-13.7, and the same conditions over-cover there
  (0.94-0.95).
- Coverage falls with lead time for persistence forecasts (about 0.89 in
  hours 1-8, 0.73-0.77 in hours 17-24). One factor per hour band (opt-in
  `--by-hour`) evens this out but leaves overall coverage at 0.81-0.84.
- One factor for patients with and without lactate in the observation
  window (opt-in `--mondrian`) keeps coverage and narrows the intervals a lot
  where some patients have none (ror_lean_exp5: x0.42-0.94); every
  full-cohort patient has lactate, so it cannot help there.

No recalibration scheme fixes a calibration/test shift on 110-190
patients; more calibration patients would. Report lactate coverage per
condition as a limitation; calibrate.py and evaluate_results.py are
unchanged.
