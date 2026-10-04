# Step 1 — `src/resolve_items.py`

This page walks through `src/resolve_items.py` from the top down: first what
the script does as a whole, then `main()`, then every function `main()` calls,
in the order it calls them. Every example uses real values from
`config/config_nibi_lean.yaml` and the MIMIC-IV 3.1 files on Nibi, where

```
$DT_REPO        = where the repo is cloned
$DT_MIMIC_DIR   = where MIMIC-IV is downloaded (the repo's mimic-iv symlink)
$DT_RESULTS_DIR = where results are recorded
```

(all three set per cluster in `jobs/setup_bash.sh`; on Nibi `$DT_RESULTS_DIR`
is the repo itself).

The MIMIC-IV dictionary tables quoted here (`d_items`, `d_labitems`) describe
*what can be measured*. They contain no patient data.

---

## `resolve_items.py` (the whole script)

**What it does:** turns the human-readable variable names in the config (e.g.
"Heart Rate") into the numeric MIMIC-IV `itemid`s that step 2 needs to pull
measurements out of `chartevents` and `labevents`.

- **Reads**
  - the config file (default `config/config.yaml`, or whatever you pass with
    `--config-file`), specifically its `paths`, `performance` and
    `variables` sections;
  - `<mimic_root>/icu/d_items.csv.gz`, the dictionary of ICU chart items;
  - `<mimic_root>/hosp/d_labitems.csv.gz`, the dictionary of lab tests.
- **Writes** `<cache_dir>/item_mapping.json`, one entry per variable holding
  the matched `itemid`s plus the full matched dictionary rows so a human can
  review them.
- **Method:** for each variable, it keeps every dictionary row whose `label`
  *contains* any of the variable's `label_patterns` (case-insensitive), or
  *equals* one if the variable sets `match: exact`. It then drops rows whose
  label contains any of its `exclude_patterns` and, for lab variables with a
  `fluids` list, rows from any other fluid. It does not hardcode itemids, so
  it adapts to whichever MIMIC-IV release is on disk.

The per-variable config keys:

| key | required | meaning |
|---|---|---|
| `name` | yes | the variable's name in the output |
| `source` | yes | `icu_chartevents` (look up in `d_items`) or `hosp_labevents` (look up in `d_labitems`) |
| `label_patterns` | yes | labels to look for |
| `match` | no | `contains` (default) or `exact` |
| `exclude_patterns` | no | drop matches whose label contains any of these |
| `fluids` | no, labs only | keep only matches whose `fluid` column is one of these, e.g. `["Blood"]` |

It does not read any patient data and finishes in a few seconds.

### Example input

Three variables from `config/config_nibi_lean.yaml`:

```yaml
variables:
  - name: heart_rate
    source: icu_chartevents
    label_patterns: ["Heart Rate"]
    exclude_patterns: ["Alarm"]
  - name: resp_rate
    source: icu_chartevents
    label_patterns: ["Respiratory Rate"]
    exclude_patterns: ["Set", "Spontaneous", "Total"]
  - name: lactate
    source: hosp_labevents
    label_patterns: ["Lactate"]
    exclude_patterns: ["Dehydrogenase"]
    fluids: ["Blood"]
```

Three rows of `icu/d_items.csv.gz`. Only the first will be kept; the other
two contain a pattern but are removed by `exclude_patterns`:

```
itemid,label,abbreviation,linksto,category,unitname,param_type,lownormalvalue,highnormalvalue
220045,Heart Rate,HR,chartevents,Routine Vital Signs,bpm,Numeric,,
220046,Heart rate Alarm - High,HR Alarm - High,chartevents,Alarms,bpm,Numeric,,
224688,Respiratory Rate (Set),Respiratory Rate (Set),chartevents,Respiratory,insp/min,Numeric,,
```

Three rows of `hosp/d_labitems.csv.gz`. Only the first will be kept:

```
itemid,label,fluid,category
50813,Lactate,Blood,Blood Gas
50954,Lactate Dehydrogenase (LD),Blood,Chemistry
51795,"Lactate Dehydrogenase, CSF",Cerebrospinal Fluid,Chemistry
```

### Example output

Three entries of `$DT_RESULTS_DIR/mimic-iv-twin-work/cache/item_mapping.json` (lactate
trimmed to its first match):

```json
{
  "heart_rate": {
    "source": "icu_chartevents",
    "itemids": [220045],
    "matches": [
      {"itemid": 220045, "label": "Heart Rate", "category": "Routine Vital Signs",
       "param_type": "Numeric", "unitname": "bpm"}
    ]
  },
  "resp_rate": {
    "source": "icu_chartevents",
    "itemids": [220210],
    "matches": [
      {"itemid": 220210, "label": "Respiratory Rate", "category": "Respiratory",
       "param_type": "Numeric", "unitname": "insp/min"}
    ]
  },
  "lactate": {
    "source": "hosp_labevents",
    "itemids": [50813, 52442, 53154],
    "matches": [
      {"itemid": 50813, "label": "Lactate", "fluid": "Blood", "category": "Blood Gas"}
    ]
  }
}
```

And the log lines:

```
[resolve_items] INFO: Resolved 'heart_rate' -> 1 itemid(s): [220045]
[resolve_items] INFO: Resolved 'resp_rate' -> 1 itemid(s): [220210]
[resolve_items] INFO: Resolved 'spo2' -> 1 itemid(s): [220277]
[resolve_items] INFO: Resolved 'map' -> 3 itemid(s): [220052, 220181, 225312]
[resolve_items] INFO: Resolved 'lactate' -> 3 itemid(s): [50813, 52442, 53154]
[resolve_items] INFO: Wrote item mapping to $DT_RESULTS_DIR/mimic-iv-twin-work/cache/item_mapping.json — REVIEW THIS FILE before running extraction.
```

> **Always review the output.** Step 2 extracts every itemid in this file
> and averages the values per hour. Nothing in step 2 checks
> `plausible_range`, so an unwanted item goes straight into the panel.
> Substring matching easily catches too much. Before `exclude_patterns`,
> `match` and `fluids` were tightened, the configs resolved:
>
> | variable | unwanted matches |
> |---|---|
> | `heart_rate`, `spo2` | alarm thresholds, "SpO2 Desat Limit", a sensor-placement checkbox |
> | `lactate` | lactate dehydrogenase (LDH), a different test in IU/L, from 4 fluids |
> | most labs | the same test in urine, ascites, pleural fluid, CSF, stool, joint fluid |
> | `hemoglobin` | carboxy-/methemoglobin, Hb A1c, fetal Hb and others (25 items) |
> | `ph` | ~130 items whose label merely contains "ph" (Phosphate, Lymphocytes, …) |
>
> If a mapping is still wrong, tighten the config and re-run step 1, or edit
> `item_mapping.json` by hand. Step 2 reads the file as-is, but re-running
> step 1 overwrites manual edits.

---

## Entry point (`if __name__ == "__main__":`)

**What it does:** reads the command line and calls `main()` with the config
path.

1. It builds a one-line description from the first line of the module
   docstring: `"Step 0: resolve the tight variable panel's item labels
   (config.yaml) to concrete"`. The docstring still calls this "Step 0"; the
   job scripts call it step 1.
2. It calls `parse_step_args()` [see below], which returns the parsed
   arguments.
3. It calls `main(args.config_file)`.

```
$ python src/resolve_items.py --config-file config/config_nibi_lean.yaml
  → args.config_file = "config/config_nibi_lean.yaml"
  → main("config/config_nibi_lean.yaml")

$ python src/resolve_items.py
  → args.config_file = "config/config.yaml"      (the default)
```

At import time the module also creates its logger with
`get_logger("resolve_items")` [see below], which is why every log line is
tagged `[resolve_items]`.

---

## `main(config_path)`

**What it does:** runs the whole workflow: load config, check folders, find
the two dictionary files, resolve every variable, write the JSON.

`config_path` arrives as `"config/config_nibi_lean.yaml"`.

1. **Load the config.** `cfg = load_config(config_path)` [see below]. `cfg`
   is a plain dict; the parts this script uses look like:

   ```python
   cfg["paths"] = {
       "mimic_root":  "$DT_REPO/mimic-iv",          # symlink -> $DT_MIMIC_DIR
       "work_dir":    "$DT_RESULTS_DIR/mimic-iv-twin-work",
       "cache_dir":   "$DT_RESULTS_DIR/mimic-iv-twin-work/cache",
       "results_dir": "$DT_RESULTS_DIR/mimic-iv-twin-work/results",
   }
   cfg["performance"] = {"duckdb_threads": 10, "duckdb_memory_limit_gb": 120, ...}
   cfg["variables"]   = [ {"name": "heart_rate", "source": "icu_chartevents",
                           "label_patterns": ["Heart Rate"], ...}, ... ]   # 5 entries
   ```

2. **Make sure the output folders exist.** `ensure_work_dirs(cfg)` [see
   below] creates `work_dir`, `cache_dir` and `results_dir` if they are
   missing.

3. **Check the MIMIC-IV layout.** `require_mimic_layout(cfg)` [see below]
   stops with a clear error if `mimic-iv/hosp` or `mimic-iv/icu` is missing.

4. **Find the ICU dictionary.** It gets the ICU folder from `icu_dir(cfg)`
   [see below] and asks `_find_file()` [see below] for the `d_items` table in
   it:

   ```python
   d_items_path = Path("$DT_REPO/mimic-iv/icu/d_items.csv.gz")
   ```

5. **Find the lab dictionary.** The same with `hosp_dir(cfg)` [see below] and
   `d_labitems`:

   ```python
   d_labitems_path = Path("$DT_REPO/mimic-iv/hosp/d_labitems.csv.gz")
   ```

6. **Open DuckDB.** `con = get_duckdb_connection(cfg)` [see below] returns an
   in-memory DuckDB connection set to 10 threads and a 120 GB memory limit.
   DuckDB queries the `.csv.gz` files in place, with no import step.

7. **Resolve each variable.** It starts with `mapping = {}` and loops over
   `cfg["variables"]`. For each variable:

   - `patterns = var["label_patterns"]`
   - `excludes = var.get("exclude_patterns", [])`. Variables without
     exclusions get an empty list.
   - `match = var.get("match", "contains")`. Anything other than `contains`
     or `exact` raises `ValueError`. `exact = (match == "exact")`.
   - It picks a resolver by `var["source"]`:
     - `"icu_chartevents"` → `resolve_icu_item(con, d_items_path, patterns, excludes, exact=exact)` [see below].
       A `fluids` key here raises `ValueError`, since `d_items` has no fluid column.
     - `"hosp_labevents"` → `resolve_lab_item(con, d_labitems_path, patterns, excludes, exact=exact, fluids=var.get("fluids"))` [see below]
     - anything else → `ValueError("Unknown source ...")`, which stops the run.
   - If nothing matched, it logs a **warning** naming the dictionary file to
     inspect by hand, and still writes an entry with an empty `itemids` list.
     Otherwise it logs the matched itemids.
   - It adds the result to `mapping` under the variable's name.

   Walking through `resp_rate`:

   ```python
   var      = {"name": "resp_rate", "source": "icu_chartevents",
               "label_patterns": ["Respiratory Rate"],
               "exclude_patterns": ["Set", "Spontaneous", "Total"], ...}
   patterns = ["Respiratory Rate"]
   excludes = ["Set", "Spontaneous", "Total"]
   match    = "contains"            # not set in the config
   exact    = False
   matches  = resolve_icu_item(...)   # → 1 row, see resolve_icu_item below
   # log: Resolved 'resp_rate' -> 1 itemid(s): [220210]
   mapping["resp_rate"] = {
       "source":  "icu_chartevents",
       "itemids": [220210],
       "matches": [{"itemid": 220210, "label": "Respiratory Rate",
                    "category": "Respiratory", "param_type": "Numeric",
                    "unitname": "insp/min"}],
   }
   ```

   After all five variables `mapping` has the keys `heart_rate`,
   `resp_rate`, `spo2`, `map`, `lactate`.

8. **Write the result.** `mapping` is written as indented JSON to
   `<cache_dir>/item_mapping.json`, overwriting any earlier version, and the
   "REVIEW THIS FILE" line is logged.

   ```
   out_path = $DT_RESULTS_DIR/mimic-iv-twin-work/cache/item_mapping.json
   ```

---

## `parse_step_args(description)` — `src/common.py`

**What it does:** defines the command line shared by all the step scripts and
checks that the config file exists.

- One option, `--config-file` (with `--config` accepted as an alias), stored
  as `args.config_file`, default `"config/config.yaml"`. Relative paths are
  resolved against the current directory, so run from the repo root (the job
  scripts `cd` there).
- If the file doesn't exist, it exits with status 2 and a usage message
  instead of a traceback:

```
$ python src/resolve_items.py --config-file nope.yaml
usage: resolve_items.py [-h] [--config-file CONFIG_FILE]
resolve_items.py: error: config file not found: nope.yaml
```

Returns: `Namespace(config_file='config/config_nibi_lean.yaml')`.

---

## `get_logger(name)` — `src/common.py`

**What it does:** returns a logger that prints to stdout (so it ends up in
the Slurm `.out` file) in the format `time [name] LEVEL: message`. The level
comes from the `TWIN_LOG_LEVEL` environment variable and defaults to `INFO`.
It adds its handler only once, so calling it repeatedly doesn't duplicate
lines.

```
get_logger("resolve_items").info("Resolved ...")
→ 2026-09-29 23:16:27,843 [resolve_items] INFO: Resolved ...
```

---

## `load_config(config_path)` — `src/common.py`

**What it does:** reads the YAML config into a dict, then makes the paths
usable.

1. Parses the YAML file with `yaml.safe_load`.
2. For **every** entry under `paths`, expands environment variables first
   (`$DT_RESULTS_DIR`, set by `jobs/setup_bash.sh`) and then `~`; a path
   that is still relative is taken relative to the repository base, not the
   current directory:

   ```
   in  : mimic_root: "mimic-iv"            work_dir: "$DT_RESULTS_DIR/mimic-iv-twin-work"
   out : "$DT_REPO/mimic-iv"              "$DT_RESULTS_DIR/mimic-iv-twin-work" (expanded)
   ```

   A variable that isn't set (the shell never sourced `jobs/setup_bash.sh`)
   stops it right away: `ValueError: paths.work_dir in config/config_nibi_lean.yaml
   uses an unset variable ... — run `source jobs/setup_bash.sh` first`.
3. If the config has no `performance` section (older configs), it fills in
   defaults: all CPU cores for DuckDB, an 8 GB DuckDB memory limit, one
   concurrent LLM request, and no batched baseline prediction.

Returns the dict shown in step 1 of `main()`.

---

## `ensure_work_dirs(cfg)` — `src/common.py`

**What it does:** creates `paths.work_dir`, `paths.cache_dir` and
`paths.results_dir`, including any missing parent folders. If a folder
already exists it does nothing.

```
creates (if missing):
  $DT_RESULTS_DIR/mimic-iv-twin-work
  $DT_RESULTS_DIR/mimic-iv-twin-work/cache
  $DT_RESULTS_DIR/mimic-iv-twin-work/results
```

Returns nothing.

---

## `require_mimic_layout(cfg)` — `src/common.py`

**What it does:** fails early if the MIMIC-IV data isn't where the config
says. It builds the two module folders with `hosp_dir(cfg)` and `icu_dir(cfg)`
[see below] and raises `FileNotFoundError` listing whichever is missing:

```
checks: $DT_REPO/mimic-iv/hosp   ✓     (mimic-iv -> $DT_MIMIC_DIR)
        $DT_REPO/mimic-iv/icu    ✓
→ returns None

if the mimic-iv symlink were missing (bash jobs/setup_bash.sh never run):
FileNotFoundError: Expected MIMIC-IV 'hosp' and 'icu' module folders not found:
['$DT_REPO/mimic-iv/hosp', '$DT_REPO/mimic-iv/icu']. This project assumes the raw
PhysioNet export is placed at $DT_REPO/mimic-iv/hosp and $DT_REPO/mimic-iv/icu ...
```

It only checks the folders exist, not which files are inside them. That is
`_find_file()`'s job.

---

## `icu_dir(cfg)` and `hosp_dir(cfg)` — `src/common.py`

**What they do:** return `paths.mimic_root` joined with the module name, as
a `Path`.

```python
icu_dir(cfg)  → Path("$DT_REPO/mimic-iv/icu")
hosp_dir(cfg) → Path("$DT_REPO/mimic-iv/hosp")
```

---

## `_find_file(directory, stem)` — `src/resolve_items.py`

**What it does:** finds a MIMIC-IV table in a folder whether it was
downloaded compressed or not. It tries `<stem>.csv.gz` first, then
`<stem>.csv`, and returns the first that exists.

```python
_find_file(Path("$DT_REPO/mimic-iv/icu"), "d_items")
  tries $DT_REPO/mimic-iv/icu/d_items.csv.gz   → exists
  → Path("$DT_REPO/mimic-iv/icu/d_items.csv.gz")

_find_file(Path("$DT_REPO/mimic-iv/hosp"), "d_labitems")
  → Path("$DT_REPO/mimic-iv/hosp/d_labitems.csv.gz")
```

If neither exists: `FileNotFoundError: Could not find d_items.csv or
d_items.csv.gz under $DT_REPO/mimic-iv/icu`.

`extract_cohort.py` has an identical copy of this function.

---

## `get_duckdb_connection(cfg)` — `src/common.py`

**What it does:** opens an in-memory DuckDB connection with the thread and
memory limits from `cfg["performance"]`, so every step shares the same
resource settings.

1. `duckdb.connect()` creates an in-memory database. Nothing is written to
   disk except spill files.
2. `PRAGMA threads=10` comes from `duckdb_threads`. Without a value it uses
   all CPU cores.
3. `PRAGMA memory_limit='120GB'` comes from `duckdb_memory_limit_gb`.
   Without a value it uses 8 GB.
4. It sets where DuckDB writes temporary data when it exceeds that limit:
   `performance.duckdb_temp_dir` if set, otherwise `$SLURM_TMPDIR` inside a
   Slurm job, plus `/duckdb_tmp`. Outside a job with neither set, DuckDB
   keeps its own default (`./.tmp`).

```
inside a job:  temp_directory = $SLURM_TMPDIR/duckdb_tmp   (node-local disk)
```

For step 1 these limits hardly matter: the two dictionary tables are tiny
(4,095 and 1,650 rows in MIMIC-IV 3.1). They matter in step 2.

Returns the connection, used as `con` below.

---

## `resolve_icu_item(con, d_items_path, patterns, excludes, exact=False)` — `src/resolve_items.py`

**What it does:** finds ICU chart items whose label contains (or, with
`exact=True`, equals) any of `patterns`, then removes those whose label
contains any of `excludes`.

1. **Build the WHERE clause** with `_label_clauses(patterns, exact)` [see
   below]. For `map` (3 patterns, contains-match):

   ```sql
   SELECT itemid, label, category, param_type, unitname
   FROM read_csv_auto(?, ignore_errors=true)
   WHERE (label ILIKE '%' || ? || '%' OR label ILIKE '%' || ? || '%' OR label ILIKE '%' || ? || '%')
   ```

   ```python
   params = ["$DT_REPO/mimic-iv/icu/d_items.csv.gz",
             "Arterial Blood Pressure mean",
             "Non Invasive Blood Pressure mean",
             "ART BP Mean"]
   ```

   `read_csv_auto` reads the gzipped CSV directly and works out the column
   types. `ignore_errors=true` skips any row it can't parse instead of
   failing.

2. **Run it.** `fetchall()` returns a list of tuples. For `resp_rate`
   (pattern `"Respiratory Rate"`):

   ```python
   rows = [
     (220210, "Respiratory Rate",               "Respiratory", "Numeric", "insp/min"),
     (224688, "Respiratory Rate (Set)",         "Respiratory", "Numeric", "insp/min"),
     (224689, "Respiratory Rate (spontaneous)", "Respiratory", "Numeric", "insp/min"),
     (224690, "Respiratory Rate (Total)",       "Respiratory", "Numeric", "insp/min"),
   ]
   ```

3. **Turn tuples into dicts** keyed by
   `["itemid", "label", "category", "param_type", "unitname"]`:

   ```python
   {"itemid": 220210, "label": "Respiratory Rate", "category": "Respiratory",
    "param_type": "Numeric", "unitname": "insp/min"}
   ```

4. **Apply exclusions** in Python, also case-insensitive and also
   "contains". A null label is treated as `""`.

   ```
   excludes = ["Set", "Spontaneous", "Total"]
   "Respiratory Rate"               → keep
   "Respiratory Rate (Set)"         → drop ("set")
   "Respiratory Rate (spontaneous)" → drop ("spontaneous")
   "Respiratory Rate (Total)"       → drop ("total")
   ```

   The exclusions are substring matches too, so a short word like `"Set"`
   would also remove any label that merely *contains* "set", such as
   "Reset" or "Offset".

Returns the list of kept dicts. For `resp_rate` that's the single 220210
row. For `heart_rate`, the query finds "Heart Rate", "Heart rate Alarm -
High" and "Heart Rate Alarm - Low", and `exclude_patterns: ["Alarm"]` leaves
only 220045.

---

## `_label_clauses(patterns, exact)` — `src/resolve_items.py`

**What it does:** builds the SQL `WHERE` condition that both resolvers use:
one clause per pattern, joined with `OR`.

- `exact=False`: `label ILIKE '%' || ? || '%'`. `ILIKE` is case-insensitive,
  and the `%` on both sides means "contains".
- `exact=True`: `label ILIKE ?`, meaning "equals, ignoring case". This is
  for short labels that turn up inside many unrelated ones: `"pH"` is
  contained in "Phosphate" and "Lymphocytes", and `"Hemoglobin"` in
  "Carboxyhemoglobin".

The `?` placeholders are filled in by DuckDB, so a pattern containing a quote
can't break the SQL.

```python
_label_clauses(["Heart Rate"], exact=False)
→ "label ILIKE '%' || ? || '%'"

_label_clauses(["Arterial Blood Pressure mean", "Non Invasive Blood Pressure mean", "ART BP Mean"], exact=False)
→ "label ILIKE '%' || ? || '%' OR label ILIKE '%' || ? || '%' OR label ILIKE '%' || ? || '%'"

_label_clauses(["pH"], exact=True)
→ "label ILIKE ?"
```

---

## `resolve_lab_item(con, d_labitems_path, patterns, excludes, exact=False, fluids=None)` — `src/resolve_items.py`

**What it does:** the same as `resolve_icu_item()`, but against
`hosp/d_labitems`, whose columns are `itemid, label, fluid, category`. It
also takes an optional `fluids` list: after the exclusions, it keeps only
rows whose `fluid` equals one of them, ignoring case.

For `lactate` (pattern `"Lactate"`, exclude `"Dehydrogenase"`, fluids
`["Blood"]`), the query returns 8 rows:

```python
rows = [
  (50813, "Lactate",                          "Blood",               "Blood Gas"),
  (50843, "Lactate Dehydrogenase, Ascites",   "Ascites",             "Chemistry"),
  (50954, "Lactate Dehydrogenase (LD)",       "Blood",               "Chemistry"),
  (51054, "Lactate Dehydrogenase, Pleural",   "Pleural",             "Chemistry"),
  (51795, "Lactate Dehydrogenase, CSF",       "Cerebrospinal Fluid", "Chemistry"),
  (51944, "Lactate Dehydrogenase, Stool",     "Stool",               "Chemistry"),
  (52442, "Lactate",                          "Blood",               "Blood Gas"),
  (53154, "Lactate",                          "Blood",               "Chemistry"),
]
```

1. **Exclusions** (label only): `"Dehydrogenase"` removes the five LDH rows
   (50843, 50954, 51054, 51795, 51944).
2. **Fluid filter:** all three remaining rows are `Blood`, so none are
   removed here.

Returns 3 dicts, the `itemids` list `[50813, 52442, 53154]`.

The fluid filter matters more for the 19-variable configs. For `sodium`
(pattern `"Sodium"`, `fluids: ["Blood"]`), the query finds 11 rows. Seven
come from other fluids: "Sodium, Ascites", "Sodium, Pleural", "Sodium, CSF",
and so on. `fluids` keeps 50824, 50983, 52455 and 52623. Filtering on the
`fluid` column works even when the label doesn't name the fluid. For
example, 51977 "Creatinine, Blood" has fluid `Urine`.

With `match: exact`, `ph` (pattern `"pH"`, `fluids: ["Blood"]`) matches only
labels that are exactly "pH": 50820 (Blood), 50831 (Other Body Fluid), 51094,
51491 and 52730 (Urine), and 52041 (Fluid). The fluid filter then leaves
50820.

---

## Where the output goes next

`src/extract_cohort.py` (step 2) loads `item_mapping.json`, collects the
`itemids` of every `icu_chartevents` variable and every `hosp_labevents`
variable, and pulls those items' measurements from `chartevents` and
`labevents`. It uses only `source` and `itemids`; `matches` is there for you
to read. A variable with an empty `itemids` list simply contributes no data.
