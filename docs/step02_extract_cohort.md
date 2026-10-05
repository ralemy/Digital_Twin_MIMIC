# Step 2 — `src/extract_cohort.py`

This page walks through `src/extract_cohort.py` from the top down: first what
the script does as a whole, then `main()`, then every function `main()` calls,
in the order it calls them. Configuration values come from
`config/config_alliance_lean.yaml` on Nibi, where

```
$DT_REPO        = where the repo is cloned
$DT_MIMIC_DIR   = where MIMIC-IV is downloaded (the repo's mimic-iv symlink)
$DT_RESULTS_DIR = where results are recorded
```

(set from your profile, `~/.config/dt_profile.yml`, by `jobs/setup_bash.sh`).

> **About the sample rows.** Every patient-level row on this page
> (`subject_id`, `stay_id`, timestamps, measured values) is **invented** for
> illustration. It has the shape of MIMIC-IV data but is not copied from it.
> Itemids, column names, file names and config values are real.

---

## `extract_cohort.py` (the whole script)

**What it does:** picks the ICU stays that make up the study cohort, pulls
their measurements for the five panel variables out of the raw MIMIC-IV
tables, turns those measurements into an hourly time series per stay, and
assigns each stay to train / val / test.

- **Reads**
  - the config file (`--config-file`), specifically its `paths`,
    `performance`, `cohort`, `variables` and `deterioration_labels`
    sections;
  - `<cache_dir>/item_mapping.json`, written by step 1
    ([resolve_items.md](resolve_items.md)), which says which `itemid`s
    belong to which variable;
  - from `<mimic_root>/icu/`: `icustays`, `chartevents`, `d_items`,
    `inputevents`;
  - from `<mimic_root>/hosp/`: `patients`, `admissions`, `labevents`.
- **Writes**
  - `<cache_dir>/chartevents_panel_48h.parquet` and
    `<cache_dir>/labevents_panel_48h.parquet`: the raw panel measurements of
    the eligible stays, first 48 hours only. These are caches, so a re-run
    skips the slow scan of `chartevents` (the biggest MIMIC-IV table).
  - `<cache_dir>/vasopressor_events_48h.parquet`: vasopressor start times in
    the first 48 hours, used later for the RQ3 "deteriorating" subgroup.
    Optional: if extraction fails, the run continues without it.
  - `<work_dir>/cohort.parquet`: one row per stay in the final cohort, with
    its split.
  - `<work_dir>/panel_long.parquet`: the hourly panel in long format, one row
    per (stay, variable, hour) that has at least one measurement.
- **Inclusion criteria**, applied in this order:
  1. ICU stay of at least 48 hours (`min_icu_stay_hours`);
  2. of those, the first one per patient (`first_stay_only: true`). Earlier,
     shorter stays are skipped, and a patient is excluded only if none of
     their stays reaches 48 hours;
  3. age ≥ 18 (`min_age_years`);
  4. at least 50% of the expected hourly panel cells populated in the first
     48 hours (`min_panel_coverage`);
  5. then, if more stays survive than `max_patients` (3000 in the lean config), a
     random sample of 3000.
- **Engine:** steps 1–3 and the raw-data scans run in DuckDB directly over the
  `.csv.gz` files. The hourly resampling, coverage filter and split run in
  pandas on the much smaller cached parquet files.

### Example input

From `config/config_alliance_lean.yaml`:

```yaml
cohort:
  min_age_years: 18
  min_icu_stay_hours: 48
  first_stay_only: true
  min_panel_coverage: 0.5
  observation_window_hours: 24
  forecast_horizon_hours: 24
  train_frac: 0.7
  val_frac: 0.15
  test_frac: 0.15
  random_seed: 42
  max_patients: 3000
deterioration_labels:
  vasopressor_itemid_labels: ["Norepinephrine", "Epinephrine", "Vasopressin", "Phenylephrine", "Dopamine"]
```

From `<cache_dir>/item_mapping.json` (only `source` and `itemids` are used;
`matches` is dropped here):

```json
{
  "heart_rate": {"source": "icu_chartevents", "itemids": [220045]},
  "resp_rate":  {"source": "icu_chartevents", "itemids": [220210]},
  "spo2":       {"source": "icu_chartevents", "itemids": [220277]},
  "map":        {"source": "icu_chartevents", "itemids": [220052, 220181, 225312]},
  "lactate":    {"source": "hosp_labevents",  "itemids": [50813, 52442, 53154]}
}
```

A stay in `icu/icustays.csv.gz` (invented):

```
subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los
10000001,20000001,30000001,MICU,MICU,2150-03-01 08:00:00,2150-03-04 10:00:00,3.08
```

A few of its rows in `icu/chartevents.csv.gz` (invented, columns trimmed):

```
subject_id,hadm_id,stay_id,charttime,itemid,value,valuenum,valueuom
10000001,20000001,30000001,2150-03-01 08:15:00,220045,92,92,bpm
10000001,20000001,30000001,2150-03-01 08:45:00,220045,96,96,bpm
10000001,20000001,30000001,2150-03-01 09:05:00,220052,71,71,mmHg
10000001,20000001,30000001,2150-03-01 09:10:00,220181,75,75,mmHg
```

### Example output

`<work_dir>/cohort.parquet` (invented):

| subject_id | hadm_id | stay_id | intime | outtime | los_hours | age_at_admission | split |
|---|---|---|---|---|---|---|---|
| 10000001 | 20000001 | 30000001 | 2150-03-01 08:00 | 2150-03-04 10:00 | 74 | 63 | train |
| 10000007 | 20000019 | 30000012 | 2161-11-20 22:30 | 2161-11-23 01:00 | 51 | 47 | test |

`<work_dir>/panel_long.parquet` (invented):

| stay_id | variable | hour | value |
|---|---|---|---|
| 30000001 | heart_rate | 0 | 94.0 |
| 30000001 | map | 1 | 73.0 |
| 30000001 | lactate | 2 | 2.1 |
| … | … | … | … |

`hour` is the whole number of hours since ICU admission (`intime`), 0 to 48.
`value` is the mean of all measurements of that variable in that hour. Hours
with no measurement have **no row**; filling gaps is left to later steps.

And the log lines. The 31487 count is from the job of 2026-09-29, which ran
the old first-stay rule (a short first stay excluded the patient). Under the
current rule the count will be somewhat higher.

```
[extract_cohort] INFO: Stays passing age/LOS/first-stay filters: 31487
[extract_cohort] INFO: Scanning chartevents for 6 itemids over 31487 stays (one-time pass, can take a while)...
[extract_cohort] INFO: Wrote .../cache/chartevents_panel_48h.parquet
[extract_cohort] INFO: Scanning labevents for 3 itemids over 31487 admissions (one-time pass)...
[extract_cohort] INFO: Wrote .../cache/labevents_panel_48h.parquet
[extract_cohort] INFO: Wrote .../cache/vasopressor_events_48h.parquet
[extract_cohort] INFO: Stays passing 50% panel-coverage threshold: N / 31487     (N not yet known)
[extract_cohort] INFO: Subsampled cohort to max_patients=3000 for this run.
[extract_cohort] INFO: Split sizes: {'train': 2100, 'val': 450, 'test': 450}
[extract_cohort] INFO: Cohort extraction complete: 3000 stays, ... panel rows. Written to $DT_RESULTS_DIR/mimic-iv-twin-work
```

---

## Entry point (`if __name__ == "__main__":`)

**What it does:** reads the command line and calls `main()` with the config
path.

1. It takes the first line of the module docstring as the `--help`
   description: `"Step 1: build the analytic cohort and extract the tight
   variable panel, entirely"`. The docstring still calls this "Step 1"; the
   job scripts call it step 2. If the docstring were missing (e.g. under
   `python -OO`), a fallback string is used instead.
2. It calls `parse_step_args()` (see [resolve_items.md](resolve_items.md#parse_step_argsdescription--srccommonpy)),
   which returns the parsed arguments and exits early if the config file
   doesn't exist.
3. It calls `main(args.config_file)`.

```
$ python src/extract_cohort.py --config-file config/config_alliance_lean.yaml
  → args.config_file = "config/config_alliance_lean.yaml"
  → main("config/config_alliance_lean.yaml")
```

At import time the module also creates its logger with
`get_logger("extract_cohort")`, which is why every log line is tagged
`[extract_cohort]`.

---

## `main(config_path)`

**What it does:** runs the whole workflow: load config and item mapping,
choose eligible stays, cache the raw measurements, resample and filter, split,
write the two output files.

`config_path` arrives as `"config/config_alliance_lean.yaml"`.

1. **Load the config.** `cfg = load_config(config_path)` (see
   [resolve_items.md](resolve_items.md#load_configconfig_path--srccommonpy))
   returns a plain dict with every path resolved (`$DT_RESULTS_DIR` expanded,
   `mimic-iv` taken relative to the repository base):

   ```python
   cfg["paths"] = {
       "mimic_root":  "$DT_REPO/mimic-iv",          # symlink -> $DT_MIMIC_DIR
       "work_dir":    "$DT_RESULTS_DIR/mimic-iv-twin-work",
       "cache_dir":   "$DT_RESULTS_DIR/mimic-iv-twin-work/cache",
       "results_dir": "$DT_RESULTS_DIR/mimic-iv-twin-work/results",
   }
   cfg["cohort"]    = {"min_age_years": 18, "min_icu_stay_hours": 48, ..., "max_patients": 3000}
   cfg["variables"] = [{"name": "heart_rate", ...}, ...]   # 5 entries
   ```

2. **Make sure the output folders exist.** `ensure_work_dirs(cfg)` creates
   `work_dir`, `cache_dir` and `results_dir` if missing.

3. **Check the MIMIC-IV layout.** `require_mimic_layout(cfg)` stops with a
   clear error if `mimic-iv/hosp` or `mimic-iv/icu` is missing.

4. **Load the item mapping from step 1.**

   ```python
   mapping_path = Path("$DT_RESULTS_DIR/mimic-iv-twin-work/cache/item_mapping.json")
   ```

   If the file is missing it raises `FileNotFoundError("... run
   src/resolve_items.py first.")`. Otherwise it parses it:

   ```python
   item_mapping = {
       "heart_rate": {"source": "icu_chartevents", "itemids": [220045], "matches": [...]},
       "resp_rate":  {"source": "icu_chartevents", "itemids": [220210], "matches": [...]},
       "spo2":       {"source": "icu_chartevents", "itemids": [220277], "matches": [...]},
       "map":        {"source": "icu_chartevents", "itemids": [220052, 220181, 225312], "matches": [...]},
       "lactate":    {"source": "hosp_labevents",  "itemids": [50813, 52442, 53154], "matches": [...]},
   }
   ```

5. **Open DuckDB.** `con = get_duckdb_connection(cfg)` (see
   [resolve_items.md](resolve_items.md#get_duckdb_connectioncfg--srccommonpy))
   returns an in-memory DuckDB connection with 10 threads and a 120 GB memory
   limit, spilling to `$SLURM_TMPDIR/duckdb_tmp` inside a Slurm job.

6. **Pick the eligible stays.** `eligible = build_eligible_stays(con, cfg)`
   [see below] returns a DataFrame with one row per stay that passes the
   first-stay, age and length-of-stay filters (31,487 on MIMIC-IV 3.1 under
   the old first-stay rule; more under the current one):

   ```
      subject_id   hadm_id   stay_id              intime             outtime  los_hours  age_at_admission
   0    10000001  20000001  30000001 2150-03-01 08:00:00 2150-03-04 10:00:00         74                63
   1    10000007  20000019  30000012 2161-11-20 22:30:00 2161-11-23 01:00:00         51                47
   ...
   ```

7. **Remember the number of variables.**
   `eligible.attrs["n_vars"] = len(cfg["variables"])`, i.e. `5`. `attrs` is
   pandas' metadata slot on a DataFrame; `resample_and_filter()` reads it
   back to compute coverage.

8. **Cache the raw panel measurements.**
   `cache_panel_raw(con, cfg, eligible, item_mapping)` [see below] writes
   `chartevents_panel_48h.parquet` and `labevents_panel_48h.parquet` (or
   reuses them if they already exist). It returns nothing.

9. **Cache vasopressor events.** `cache_vasopressor_events(con, cfg, eligible)`
   [see below] writes `vasopressor_events_48h.parquet` (or reuses it, or skips
   it with a warning). It returns nothing.

10. **Close DuckDB.** `con.close()` releases DuckDB's memory before pandas
    starts working, so the two phases don't hold memory at the same time.

11. **Resample and apply the coverage filter.**
    `cohort, panel_long = resample_and_filter(cfg, eligible, item_mapping)`
    [see below] returns the final cohort (≤ 3000 rows, same columns as
    `eligible`) and its hourly panel:

    ```
    cohort:     3000 rows × [subject_id, hadm_id, stay_id, intime, outtime, los_hours, age_at_admission]
    panel_long: rows × [stay_id, variable, hour, value]
    ```

12. **Assign splits.** `cohort = assign_splits(cohort, cfg)` [see below] adds
    a `split` column with `"train"`, `"val"` or `"test"`.

13. **Write outputs.** Both DataFrames are written without the pandas index:

    ```
    $DT_RESULTS_DIR/mimic-iv-twin-work/cohort.parquet
    $DT_RESULTS_DIR/mimic-iv-twin-work/panel_long.parquet
    ```

    and the "Cohort extraction complete" line is logged.

---

## `build_eligible_stays(con, cfg)`

**What it does:** one SQL query that applies the first-stay, length-of-stay
and age criteria and returns the surviving stays as a DataFrame.

1. **Find the three input files** with `_find_file()` [see below]:

   ```python
   icustays_path   = Path("$DT_REPO/mimic-iv/icu/icustays.csv.gz")
   patients_path   = Path("$DT_REPO/mimic-iv/hosp/patients.csv.gz")
   admissions_path = Path("$DT_REPO/mimic-iv/hosp/admissions.csv.gz")
   ```

2. **Build the first-stay clause.** With `first_stay_only: true`,
   `first_stay_clause = "AND rn = 1"`. With `false` it would be `""` and every
   stay of a patient could enter the cohort.

3. **Build the query.** It has three parts (CTEs) and a final `SELECT`:

   - `stays` reads `icustays`, **keeps only stays of at least 48 hours**
     (its `WHERE` clause), and adds two columns:
     - `los_hours = date_diff('hour', intime, outtime)`, the length of stay
       in whole hours;
     - `rn = row_number() OVER (PARTITION BY subject_id ORDER BY intime)`,
       which numbers each patient's *remaining* stays 1, 2, 3 … in time
       order.

     In SQL, `WHERE` runs before window functions, so short stays are gone
     before `rn` is counted:

     ```
     subject_id  stay_id   intime               outtime              los_hours  rn
     10000001    30000002  2150-01-10 06:00:00  2150-01-11 02:00:00  20         –   ← removed by WHERE (< 48 h)
     10000001    30000001  2150-03-01 08:00:00  2150-03-04 10:00:00  74         1   ← kept: first stay ≥ 48 h
     10000001    30000044  2151-06-10 14:00:00  2151-06-15 09:00:00  115        2   ← dropped by rn = 1
     10000003    30000005  2140-01-02 03:00:00  2140-01-03 01:00:00  22         –   ← removed by WHERE; patient has no other stay, so excluded
     ```

   - `pts` reads `patients` and keeps `subject_id, anchor_age, anchor_year`.
   - `adm` reads `admissions` and keeps `hadm_id, admittime`.
   - The final `SELECT` joins the three and keeps rows with
     `anchor_age >= 18` and (here) `rn = 1`. The two numbers (48 and 18) are
     pasted into the SQL from `cfg["cohort"]`; the three file paths are
     passed as `?` parameters.

   `read_csv_auto(..., ignore_errors=true)` lets DuckDB guess column types
   and skip rows it can't parse, instead of failing.

   Notes on what the query actually does:
   - **First qualifying stay.** Because the 48-hour filter comes before
     `rn = 1`, a patient whose first ICU stay was short still enters the
     cohort with their first stay of 48 hours or more. Only patients with no
     such stay are excluded.
   - **Age is `anchor_age`**, the patient's age in their MIMIC-IV
     `anchor_year`, not the exact age at this admission. It is renamed
     `age_at_admission` in the output. MIMIC-IV shifts dates per patient, so
     `anchor_age` is the usual adult filter.
   - The `adm` join only removes stays whose `hadm_id` is not in
     `admissions`; `admittime` and `anchor_year` are not used.

4. **Run it.** `con.execute(query, [3 paths]).fetchdf()` returns a pandas
   DataFrame:

   ```
   columns: subject_id, hadm_id, stay_id, intime, outtime, los_hours, age_at_admission
   rows:    31487   (old rule; higher under the current one)
   ```

   It logs `Stays passing age/LOS/first-stay filters: <rows>` and returns
   the DataFrame.

---

## `_find_file(directory, stem)`

**What it does:** finds a MIMIC-IV table whether it is gzipped or not.

It tries `<directory>/<stem>.csv.gz` first, then `<directory>/<stem>.csv`,
and returns the first that exists. If neither does, it raises
`FileNotFoundError`.

```python
_find_file(Path("$DT_REPO/mimic-iv/icu"), "icustays")
# → Path("$DT_REPO/mimic-iv/icu/icustays.csv.gz")

_find_file(Path("$DT_REPO/mimic-iv/icu"), "nope")
# → FileNotFoundError: Could not find nope.csv or nope.csv.gz under $DT_REPO/mimic-iv/icu
```

It only checks that the file exists. A truncated or corrupted `.csv.gz`
passes here and fails later inside DuckDB (`IO Error: Input is not a GZIP
stream`, as in job 22952698).

---

## `cache_panel_raw(con, cfg, eligible_stays, item_mapping)`

**What it does:** scans `chartevents` and `labevents` once each, keeps only
the panel items of the eligible stays inside the first 48 hours, and writes
the result to parquet. If a cache file already exists, that scan is skipped.

1. **Work out the file names and window.**

   ```python
   chart_out, lab_out = _panel_cache_paths(cfg)   # see below
   # chart_out = .../cache/chartevents_panel_48h.parquet
   # lab_out   = .../cache/labevents_panel_48h.parquet
   total_h  = _window_hours(cfg)                  # 24 + 24 = 48
   stay_ids = eligible_stays["stay_id"].tolist()  # 31487 ints, used only for logging
   n_hadm   = eligible_stays["hadm_id"].nunique() # 31487, used only for logging
   ```

2. **Expose the eligible stays to DuckDB.**
   `con.register("eligible_stays_df", eligible_stays[["stay_id", "hadm_id", "intime"]])`
   makes the pandas DataFrame queryable in SQL as the table
   `eligible_stays_df`, without copying it.

3. **Split the itemids by source.** It loops over `item_mapping.values()`:

   ```python
   chart_itemids = [220045, 220210, 220277, 220052, 220181, 225312]   # icu_chartevents
   lab_itemids   = [50813, 52442, 53154]                              # hosp_labevents
   ```

4. **chartevents** (only if `chart_itemids` is non-empty):
   - Finds `$DT_REPO/mimic-iv/icu/chartevents.csv.gz` with `_find_file()`.
   - If `chart_out` already exists, logs `Reusing cached ...` and moves on.
   - Otherwise registers the itemids as table `chart_itemids_df` and runs a
     `COPY (SELECT ...) TO '<chart_out>' (FORMAT PARQUET)`, which streams the
     result straight into the parquet file. A `chartevents` row is kept if:
     - its `itemid` is one of the panel itemids,
     - `valuenum` (the numeric value) is not null, and
     - its `stay_id` is an eligible stay **and** its `charttime` is between
       `intime` and `intime + 48 hours`, inclusive.

     Only four columns are written:

     ```
     stay_id   itemid  charttime            valuenum
     30000001  220045  2150-03-01 08:15:00  92.0
     30000001  220045  2150-03-01 08:45:00  96.0
     30000001  220052  2150-03-01 09:05:00  71.0
     30000001  220181  2150-03-01 09:10:00  75.0
     ```

     A row at `2150-03-03 09:00:00` (49 h after `intime`) would be dropped.

5. **labevents** (only if `lab_itemids` is non-empty): the same, with three
   differences:
   - `labevents` has no `stay_id`, so it matches on **`hadm_id`** (the
     hospital admission) and uses that stay's `intime` for the window.
   - The output column is `hadm_id` instead of `stay_id`.
   - It writes `lab_out`.

   ```
   hadm_id   itemid  charttime            valuenum
   20000001  50813   2150-03-01 10:20:00  2.1
   20000001  50813   2150-03-01 16:05:00  1.6
   ```

   A lactate drawn in the emergency department before ICU admission
   (`charttime < intime`) is dropped.

The function returns nothing; its effect is the two files.

**Caching caveat.** The cache file name only changes with the window length.
If you change `item_mapping.json` or the `cohort` age/LOS/first-stay
settings, an existing cache is reused as-is: removed itemids are filtered out
later, but newly added itemids or newly eligible stays will be missing.
Delete the two `*_panel_48h.parquet` files to force a fresh scan.

---

## `_panel_cache_paths(cfg)`

**What it does:** returns the two cache file paths, with the window length in
the name so a different window never reuses a cache that's too short.

```python
_panel_cache_paths(cfg)
# → (Path(".../cache/chartevents_panel_48h.parquet"),
#    Path(".../cache/labevents_panel_48h.parquet"))
```

With `observation_window_hours: 36` the names would end in `_60h.parquet`.

---

## `_window_hours(cfg)`

**What it does:** observation window plus forecast horizon.

```python
_window_hours(cfg)   # 24 + 24 → 48
```

---

## `cache_vasopressor_events(con, cfg, eligible_stays)`

**What it does:** finds the itemids of the five vasopressors in `d_items`,
then extracts the `inputevents` rows (drug administrations) for those
itemids and the eligible stays that start within the first 48 hours of the
stay. Step 4 (`evaluate_results.py`) uses it to flag "deteriorating" stays
for RQ3. RQ1/RQ2 don't need it, so **nothing in this function can stop the
run**: every failure is logged as a warning and the function returns.

1. **Reuse the cache if present.**
   `out_path = vasopressor_cache_path(cfg)` [see below], i.e.
   `.../cache/vasopressor_events_48h.parquet`. If it exists, it logs
   `Reusing cached ...` and returns.

2. **Find the input files.** `d_items` and `inputevents` in `icu/`. If either
   is missing, it logs `Skipping vasopressor extraction (RQ3 subgroup will
   be unavailable): ...` and returns.

3. **Resolve vasopressor itemids.**

   ```python
   labels = ["Norepinephrine", "Epinephrine", "Vasopressin", "Phenylephrine", "Dopamine"]
   like_clauses = "label ILIKE '%' || ? || '%' OR label ILIKE '%' || ? || '%' OR ..."   # 5 clauses
   ```

   The query `SELECT DISTINCT itemid FROM d_items WHERE <like_clauses>` is a
   case-insensitive *contains* match on each label, with the label values
   passed as parameters. On MIMIC-IV 3.1 it matches 10 items (order may
   vary):

   ```
   221906 Norepinephrine          221289 Epinephrine        229617 Epinephrine.
   222315 Vasopressin             221662 Dopamine
   221749 Phenylephrine           229630 Phenylephrine (50/250)
   229631 Phenylephrine (200/250)_OLD_1    229632 Phenylephrine (200/250)
   229789 Phenylephrine (Intubation)       ← a chartevents item, never found in inputevents
   ```

   `"Epinephrine"` also matches `"Norepinephrine"`, which is harmless here
   because both are wanted. If the list is empty, it logs a warning and
   returns.

4. **Extract the events.** `window_h = vasopressor_window_hours(cfg)` [see
   below], 24 + 24 = 48. It registers the itemids as `vaso_itemids_df` and
   the eligible stays' `stay_id, intime` as `vaso_stays_df`, then `COPY`s
   `stay_id, itemid, starttime` from `inputevents` to `out_path` for rows
   where:
   - `itemid` is a vasopressor itemid, and
   - `stay_id` is an eligible stay **and** `starttime` is between `intime`
     and `intime + 48 hours`, inclusive.

   For a stay with `intime = 2161-11-20 22:30:00` (invented):

   ```
   stay_id   itemid  starttime
   30000012  221906  2161-11-20 23:15:00   ← hour 0.75, kept
   30000012  221906  2161-11-21 09:30:00   ← hour 11, kept
   30000012  221906  2161-11-22 22:30:00   ← hour 48, kept (inclusive)
   30000012  221906  2161-11-23 00:10:00   ← hour 49.7, dropped
   ```

   The window starts at `intime`, not at the observation cutoff (hour 24),
   so vasopressors given during the observation window stay in the file.
   `evaluate_results.py` counts only starts in hours 24–48, but the earlier
   rows let you tell a new vasopressor from one that was already running.

5. **Any other error** (a corrupt `.csv.gz`, an unexpected column type, a
   DuckDB error, a bad config key) is caught. It deletes any half-written
   `out_path`, so the next run doesn't reuse it as a valid cache, and logs:

   ```
   [extract_cohort] WARNING: Vasopressor extraction failed (RQ3 subgroup will be unavailable): IOException: IO Error: Input is not a GZIP stream
   ```

   `main()` then carries on with resampling as usual. `evaluate_results.py`
   finds no vasopressor file and skips only the RQ3 subgroup analysis.

---

## `vasopressor_window_hours(cfg)` and `vasopressor_cache_path(cfg)` — `src/common.py`

**What they do:** define the vasopressor window and the cache file name in
one place, so `extract_cohort.py` (which writes the file) and
`evaluate_results.py` (which reads it) always agree.

```python
vasopressor_window_hours(cfg)   # observation_window_hours + lookahead_hours = 24 + 24 → 48
vasopressor_cache_path(cfg)     # → Path(".../cache/vasopressor_events_48h.parquet")
```

The window is in the file name for the same reason as the panel caches:
with `lookahead_hours: 36` the file would be `vasopressor_events_60h.parquet`,
so a cache built for a shorter window is never reused.

---

## `resample_and_filter(cfg, eligible_stays, item_mapping)`

**What it does:** turns the raw cached measurements into an hourly panel,
drops stays with too little data, optionally subsamples to `max_patients`,
and returns `(cohort, panel_long)`. Pure pandas; DuckDB is closed by now.

1. **Window sizes.** `obs_h = 24`, `hor_h = 24`, `total_h = 48`.

2. **Reverse the mapping** so each itemid points to its variable name:

   ```python
   itemid_to_var = {220045: "heart_rate", 220210: "resp_rate", 220277: "spo2",
                    220052: "map", 220181: "map", 225312: "map",
                    50813: "lactate", 52442: "lactate", 53154: "lactate"}
   ```

   The three MAP itemids (arterial line, non-invasive cuff, a second
   arterial label) all become the same variable `map`.

3. **Index the stays by `stay_id`.**
   `stays_idx = eligible_stays.set_index("stay_id")[["intime"]]`, used to
   look up each stay's admission time.

4. **Chart events** (if `chartevents_panel_48h.parquet` exists):

   ```python
   ce = pd.read_parquet(chart_path)
   #   stay_id   itemid  charttime            valuenum
   #   30000001  220045  2150-03-01 08:15:00  92.0
   #   30000001  220052  2150-03-01 09:05:00  71.0

   ce["variable"] = ce["itemid"].map(itemid_to_var)   # "heart_rate", "map"
   ce = ce.dropna(subset=["variable"])                # drops itemids no longer in the mapping
   ce = ce.join(stays_idx, on="stay_id")              # adds intime = 2150-03-01 08:00:00
   ce["hour"] = (charttime - intime) in hours, as float
   #   08:15 → 0.25,  09:05 → 1.0833
   frames.append(ce[["stay_id", "variable", "hour", "valuenum"]])
   ```

5. **Lab events** (if `labevents_panel_48h.parquet` exists): the same, but
   the lab rows only carry `hadm_id`, so they are joined to
   `eligible_stays.set_index("hadm_id")[["stay_id", "intime"]]` to get both
   the stay and its `intime`. Rows whose `hadm_id` has no eligible stay are
   dropped.

   ```
   hadm_id 20000001, 50813, 2150-03-01 10:20:00, 2.1
     → stay_id 30000001, variable "lactate", hour 2.333, valuenum 2.1
   ```

   With `first_stay_only: true` each `hadm_id` has at most one eligible stay.
   If it were `false`, two stays in the same admission would make
   `set_index("hadm_id")` non-unique and duplicate lab rows.

   `itemid_to_var_lab` is just a copy of `itemid_to_var`.

6. **Nothing extracted?** If neither cache file exists, it raises
   `RuntimeError("No panel data extracted at all ...")`.

7. **Combine and bin.**

   ```python
   raw = pd.concat(frames)                              # all variables, long format
   raw = raw[(raw["hour"] >= 0) & (raw["hour"] <= 48)]  # same window as the SQL
   raw["hour_bin"] = floor(hour)                        # 0.25 → 0, 1.0833 → 1, 2.333 → 2, 48.0 → 48
   ```

8. **Average per hour.** Group by `(stay_id, variable, hour_bin)` and take the
   mean of `valuenum`; rename `hour_bin → hour`, `valuenum → value`:

   ```
   input (raw):                                   output (panel):
   30000001 heart_rate 0.25  92.0                 30000001 heart_rate 0  94.0
   30000001 heart_rate 0.75  96.0                 30000001 map        1  73.0
   30000001 map        1.083 71.0                 30000001 lactate    2   2.1
   30000001 map        1.167 75.0
   30000001 lactate    2.333  2.1
   ```

   Note that the two MAP readings (one invasive, one cuff) in hour 1 are
   averaged together, as are the hourly readings of any one item.

9. **Coverage.** For each stay, the number of `(variable, hour)` cells it has
   divided by the number expected:

   ```python
   n_vars        = eligible_stays.attrs.get("n_vars", len(item_mapping))   # 5 (set in main)
   expected_bins = 48 * 5                                                  # 240
   coverage      = panel.groupby("stay_id").size() / 240
   # stay 30000001 with 200 cells → 0.833  (kept)
   # stay 30000099 with  90 cells → 0.375  (dropped, < 0.5)
   keep_stays    = stay_ids with coverage >= 0.5
   ```

   Stays with no panel rows at all don't appear in `coverage` and are dropped
   too. Lactate is measured a few times a day at most, so in practice the
   four hourly vitals carry most of the coverage. Hours run 0–48 (49 bins), so
   a stay with every cell filled scores 245/240 ≈ 1.02.

   Logs `Stays passing 50% panel-coverage threshold: N / 31487`.

10. **Build the cohort.** Keep the rows of `eligible_stays` whose `stay_id`
    is in `keep_stays`. If more than `max_patients` (3000) remain, take a
    random sample of 3000 with `random_state=42` (same sample every run) and
    log it. With `max_patients: null` nothing is sampled.

11. **Trim the panel** to the stays in the cohort and return
    `(cohort, panel)`.

---

## `assign_splits(cohort, cfg)`

**What it does:** shuffles the cohort's stays with a fixed seed and assigns
the first 70% to train, the next 15% to val and the rest to test. One
patient contributes one stay, so there is no patient overlap between splits.

1. **Shuffle.** `cohort["stay_id"].sample(frac=1.0, random_state=42)` returns
   all stay ids in a random but repeatable order.

2. **Sizes**, rounding down:

   ```python
   n       = 3000
   n_train = int(3000 * 0.7)    # 2100
   n_val   = int(3000 * 0.15)   # 450
   # test  = 3000 - 2100 - 450 = 450
   ```

   `test_frac` is not read; test gets whatever is left. With `n = 2999`:
   train 2099, val 449, test 451.

3. **Assign.** Walks the shuffled list: positions 0–2099 → `"train"`,
   2100–2549 → `"val"`, 2550–2999 → `"test"`, stored in `split_map`
   (`{30000001: "train", 30000012: "test", ...}`).

4. **Add the column.** Copies `cohort`, adds
   `split = stay_id.map(split_map)`, logs
   `Split sizes: {'train': 2100, 'val': 450, 'test': 450}` and returns it.

---

## Config settings that are not used

These appear in the `cohort` section but this script doesn't read them:

| key | what happens instead |
|---|---|
| `resample_freq_minutes` | binning is always hourly (`np.floor(hour)`) |
| `test_frac` | test gets the remainder after train and val |

---

## Where the output goes next

- `src/run_experiment.py` loads `cohort.parquet` and `panel_long.parquet`,
  and `harmonization_agent.build_tensors()` turns each stay's long-format
  rows into the hour × variable grid the forecasters use.
- `src/evaluate_results.py` reads `vasopressor_events_48h.parquet` (via `vasopressor_cache_path()`) together with
  `cohort.parquet` to build the RQ3 deterioration subgroup, and skips that
  analysis with a warning if the file is missing.
- `src/smoke_test.py` writes synthetic files with the same names and
  columns, so the rest of the pipeline can be tested without MIMIC-IV.
