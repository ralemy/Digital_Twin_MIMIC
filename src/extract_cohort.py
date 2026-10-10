"""
Step 1: build the analytic cohort and extract the tight variable panel, entirely
locally with DuckDB running directly over the raw PhysioNet .csv/.csv.gz files
under $DT_MIMIC_DIR/{hosp,icu} (the repo's mimic-iv link). No data leaves the machine at any point.

Inclusion criteria (Chapter 5, Section 5.3):
  - first ICU stay per patient that lasted >= 48 hours (earlier, shorter
    stays are skipped; patients with no such stay are excluded)
  - age >= 18 at admission
  - >=50% of the tight panel's expected measurements populated in the
    observation + horizon window

Outputs (written under <work_dir>):
  cache/chartevents_panel_<N>h.parquet  filtered raw chartevents rows for the panel itemids,
                                        restricted to the first N = observation + horizon hours
                                        of each stay
  cache/labevents_panel_<N>h.parquet    filtered raw labevents rows (lactate), same window
  cache/vasopressor_events_<M>h.parquet  filtered inputevents rows used for RQ3 labels,
                                        restricted to the first M = observation + lookahead
                                        hours of each stay
  cohort.parquet                    one row per eligible stay_id, with split assignment
  panel_long.parquet                long-format (stay_id, variable, hour, value) resampled panel

Usage:
    python src/extract_cohort.py --config-file config/config_local.yaml
Requires resolve_items.py to have been run first (needs cache/item_mapping.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from common import (
    ensure_work_dirs,
    get_duckdb_connection,
    get_logger,
    hosp_dir,
    icu_dir,
    load_config,
    parse_step_args,
    require_mimic_layout,
    vasopressor_cache_path,
    vasopressor_window_hours,
)
from harmonization_agent import drop_invalid_values

log = get_logger("extract_cohort")


def _find_file(directory: Path, stem: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.csv.gz under {directory}")


def _window_hours(cfg: dict) -> int:
    c = cfg["cohort"]
    return c["observation_window_hours"] + c["forecast_horizon_hours"]


def _panel_cache_paths(cfg: dict) -> tuple[Path, Path]:
    """The panel caches only hold rows inside the observation + horizon
    window, so the window is part of the filename — changing it in the config
    triggers a fresh extraction instead of silently reusing a narrower cache."""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    total_h = _window_hours(cfg)
    return (cache_dir / f"chartevents_panel_{total_h}h.parquet",
            cache_dir / f"labevents_panel_{total_h}h.parquet")


def build_eligible_stays(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    icustays_path = _find_file(icu_dir(cfg), "icustays")
    patients_path = _find_file(hosp_dir(cfg), "patients")
    admissions_path = _find_file(hosp_dir(cfg), "admissions")

    c = cfg["cohort"]
    first_stay_clause = "AND rn = 1" if c["first_stay_only"] else ""

    # The LOS filter sits inside `stays` (WHERE runs before the window
    # function), so rn = 1 picks each patient's first stay of >= min hours
    # rather than dropping patients whose very first stay was too short.
    query = f"""
        WITH stays AS (
            SELECT
                s.subject_id, s.hadm_id, s.stay_id,
                s.intime, s.outtime,
                date_diff('hour', s.intime, s.outtime) AS los_hours,
                row_number() OVER (PARTITION BY s.subject_id ORDER BY s.intime) AS rn
            FROM read_csv_auto(?, ignore_errors=true) s
            WHERE date_diff('hour', s.intime, s.outtime) >= {c['min_icu_stay_hours']}
        ),
        pts AS (
            SELECT subject_id, anchor_age, anchor_year
            FROM read_csv_auto(?, ignore_errors=true)
        ),
        adm AS (
            SELECT hadm_id, admittime
            FROM read_csv_auto(?, ignore_errors=true)
        )
        SELECT
            st.subject_id, st.hadm_id, st.stay_id, st.intime, st.outtime, st.los_hours,
            p.anchor_age AS age_at_admission
        FROM stays st
        JOIN pts p USING (subject_id)
        JOIN adm a USING (hadm_id)
        WHERE p.anchor_age >= {c['min_age_years']}
          {first_stay_clause}
        ORDER BY st.stay_id
    """
    df = con.execute(query, [str(icustays_path), str(patients_path), str(admissions_path)]).fetchdf()
    log.info("Stays passing age/LOS/first-stay filters: %d", len(df))
    return df


def cache_panel_raw(con: duckdb.DuckDBPyConnection, cfg: dict, eligible_stays: pd.DataFrame, item_mapping: dict) -> None:
    """One filtered pass over chartevents and labevents; cached to parquet so
    later steps (and re-runs) never re-scan the full raw tables again.

    Filters are pushed down via registered temp tables (semi-joins) rather
    than string-interpolated IN (...) lists, so this stays correct and fast
    even when the eligible cohort or itemid set is large. The time window
    (intime .. intime + observation + horizon hours, inclusive — the same
    bounds resample_and_filter() applies) is pushed down too, so rows the
    later steps would discard are never written or loaded into pandas."""
    chart_out, lab_out = _panel_cache_paths(cfg)
    total_h = _window_hours(cfg)
    stay_ids = eligible_stays["stay_id"].tolist()
    n_hadm = eligible_stays["hadm_id"].nunique()

    con.register("eligible_stays_df", eligible_stays[["stay_id", "hadm_id", "intime"]])

    chart_itemids = []
    lab_itemids = []
    for info in item_mapping.values():
        if info["source"] == "icu_chartevents":
            chart_itemids.extend(info["itemids"])
        elif info["source"] == "hosp_labevents":
            lab_itemids.extend(info["itemids"])

    if chart_itemids:
        chartevents_path = _find_file(icu_dir(cfg), "chartevents")
        out_path = chart_out
        if not out_path.exists():
            log.info("Scanning chartevents for %d itemids over %d stays (one-time pass, can take a while)...",
                      len(chart_itemids), len(stay_ids))
            con.register("chart_itemids_df", pd.DataFrame({"itemid": chart_itemids}))
            con.execute(f"""
                COPY (
                    SELECT ce.stay_id, ce.itemid, ce.charttime, ce.valuenum
                    FROM read_csv_auto('{chartevents_path}', ignore_errors=true) ce
                    WHERE ce.itemid IN (SELECT itemid FROM chart_itemids_df)
                      AND ce.valuenum IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM eligible_stays_df es
                          WHERE es.stay_id = ce.stay_id
                            AND CAST(ce.charttime AS TIMESTAMP) >= es.intime
                            AND CAST(ce.charttime AS TIMESTAMP) <= es.intime + INTERVAL {total_h} HOUR
                      )
                ) TO '{out_path}' (FORMAT PARQUET)
            """)
            log.info("Wrote %s", out_path)
        else:
            log.info("Reusing cached %s", out_path)

    if lab_itemids:
        labevents_path = _find_file(hosp_dir(cfg), "labevents")
        out_path = lab_out
        if not out_path.exists():
            log.info("Scanning labevents for %d itemids over %d admissions (one-time pass)...",
                      len(lab_itemids), n_hadm)
            con.register("lab_itemids_df", pd.DataFrame({"itemid": lab_itemids}))
            con.execute(f"""
                COPY (
                    SELECT le.hadm_id, le.itemid, le.charttime, le.valuenum
                    FROM read_csv_auto('{labevents_path}', ignore_errors=true) le
                    WHERE le.itemid IN (SELECT itemid FROM lab_itemids_df)
                      AND le.valuenum IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM eligible_stays_df es
                          WHERE es.hadm_id = le.hadm_id
                            AND CAST(le.charttime AS TIMESTAMP) >= es.intime
                            AND CAST(le.charttime AS TIMESTAMP) <= es.intime + INTERVAL {total_h} HOUR
                      )
                ) TO '{out_path}' (FORMAT PARQUET)
            """)
            log.info("Wrote %s", out_path)
        else:
            log.info("Reusing cached %s", out_path)


def cache_vasopressor_events(con: duckdb.DuckDBPyConnection, cfg: dict, eligible_stays: pd.DataFrame) -> None:
    """Best-effort extraction of vasopressor administration events for the RQ3
    deterioration label. Not required for RQ1/RQ2; any failure here is logged
    and skipped rather than fatal.

    Only starts between intime and intime + observation + lookahead hours
    (inclusive) are kept: the latest point evaluate_results.py looks at.
    Starting at intime rather than at the observation cutoff keeps earlier
    vasopressor use available, e.g. to tell new starts from ongoing ones."""
    out_path = vasopressor_cache_path(cfg)
    if out_path.exists():
        log.info("Reusing cached %s", out_path)
        return
    try:
        d_items_path = _find_file(icu_dir(cfg), "d_items")
        inputevents_path = _find_file(icu_dir(cfg), "inputevents")
    except FileNotFoundError as e:
        log.warning("Skipping vasopressor extraction (RQ3 subgroup will be unavailable): %s", e)
        return

    try:
        labels = cfg["deterioration_labels"]["vasopressor_itemid_labels"]
        like_clauses = " OR ".join(["label ILIKE '%' || ? || '%'" for _ in labels])
        vaso_itemids = con.execute(
            f"SELECT DISTINCT itemid FROM read_csv_auto(?, ignore_errors=true) WHERE {like_clauses}",
            [str(d_items_path)] + labels,
        ).fetchdf()["itemid"].tolist()

        if not vaso_itemids:
            log.warning("No vasopressor itemids resolved; RQ3 subgroup will be unavailable.")
            return

        window_h = vasopressor_window_hours(cfg)
        con.register("vaso_itemids_df", pd.DataFrame({"itemid": vaso_itemids}))
        con.register("vaso_stays_df", eligible_stays[["stay_id", "intime"]])
        con.execute(f"""
            COPY (
                SELECT ie.stay_id, ie.itemid, ie.starttime
                FROM read_csv_auto('{inputevents_path}', ignore_errors=true) ie
                WHERE ie.itemid IN (SELECT itemid FROM vaso_itemids_df)
                  AND EXISTS (
                      SELECT 1 FROM vaso_stays_df vs
                      WHERE vs.stay_id = ie.stay_id
                        AND CAST(ie.starttime AS TIMESTAMP) >= vs.intime
                        AND CAST(ie.starttime AS TIMESTAMP) <= vs.intime + INTERVAL {window_h} HOUR
                  )
            ) TO '{out_path}' (FORMAT PARQUET)
        """)
        log.info("Wrote %s", out_path)
    except Exception as e:
        # A half-written file would be reused as a valid cache on the next run.
        out_path.unlink(missing_ok=True)
        log.warning("Vasopressor extraction failed (RQ3 subgroup will be unavailable): %s: %s",
                    type(e).__name__, e)


def resample_and_filter(cfg: dict, eligible_stays: pd.DataFrame, item_mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample raw events to hourly bins per stay/variable, compute panel
    coverage over the observation+horizon window, and drop stays below the
    coverage threshold. Returns (cohort_df, panel_long_df)."""
    c = cfg["cohort"]
    obs_h, hor_h = c["observation_window_hours"], c["forecast_horizon_hours"]
    total_h = obs_h + hor_h

    frames = []
    chart_path, lab_path = _panel_cache_paths(cfg)

    itemid_to_var = {}
    for var, info in item_mapping.items():
        for iid in info["itemids"]:
            itemid_to_var[iid] = var

    stays_idx = eligible_stays.set_index("stay_id")[["intime"]]

    if chart_path.exists():
        ce = pd.read_parquet(chart_path)
        ce["variable"] = ce["itemid"].map(itemid_to_var)
        ce = ce.dropna(subset=["variable"])
        ce = ce.join(stays_idx, on="stay_id")
        ce["hour"] = ((pd.to_datetime(ce["charttime"]) - pd.to_datetime(ce["intime"])).dt.total_seconds() / 3600).astype(float)
        frames.append(ce[["stay_id", "variable", "hour", "valuenum"]])

    if lab_path.exists():
        le = pd.read_parquet(lab_path)
        itemid_to_var_lab = {iid: v for iid, v in itemid_to_var.items()}
        le["variable"] = le["itemid"].map(itemid_to_var_lab)
        le = le.dropna(subset=["variable"])
        hadm_to_stay = eligible_stays.set_index("hadm_id")[["stay_id", "intime"]]
        le = le.join(hadm_to_stay, on="hadm_id")
        le = le.dropna(subset=["stay_id"])
        le["hour"] = ((pd.to_datetime(le["charttime"]) - pd.to_datetime(le["intime"])).dt.total_seconds() / 3600).astype(float)
        frames.append(le[["stay_id", "variable", "hour", "valuenum"]])

    if not frames:
        raise RuntimeError("No panel data extracted at all — check item_mapping.json and MIMIC-IV file locations.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["hour"] >= 0) & (raw["hour"] <= total_h)]
    # Charting artefacts (MAP 79104, SpO2 10099, heart rate 0) are dropped per
    # raw value, before the hourly mean would blend them with real readings.
    raw = drop_invalid_values(raw, cfg, value_col="valuenum")
    raw["hour_bin"] = np.floor(raw["hour"]).astype(int)

    panel = (
        raw.groupby(["stay_id", "variable", "hour_bin"])["valuenum"]
        .mean()
        .reset_index()
        .rename(columns={"hour_bin": "hour", "valuenum": "value"})
    )

    n_eligible = eligible_stays["stay_id"].nunique()
    if c.get("coverage_rule", "all") == "vitals_plus_labs":
        # Labs are drawn once or twice a day, so pooled hourly coverage over a
        # lab-heavy panel can't reach the threshold (19 variables: ~39% at
        # best). Apply the threshold to the hourly vitals only, and require
        # every lab at least once in the observation window.
        sources = {v["name"]: v.get("source") for v in cfg["variables"]}
        vitals = [v for v, s in sources.items() if s == "icu_chartevents" and v in item_mapping]
        labs = [v for v, s in sources.items() if s == "hosp_labevents" and v in item_mapping]
        vit = panel[panel["variable"].isin(vitals)]
        coverage = vit.groupby("stay_id").size() / (total_h * len(vitals))
        pass_vitals = set(coverage[coverage >= c["min_panel_coverage"]].index)
        obs_labs = panel[panel["variable"].isin(labs) & (panel["hour"] < obs_h)]
        n_labs_seen = obs_labs.groupby("stay_id")["variable"].nunique()
        pass_labs = set(n_labs_seen[n_labs_seen == len(labs)].index)
        keep_stays = pd.Index(sorted(pass_vitals & pass_labs))
        log.info("Coverage rule vitals_plus_labs: %d / %d stays reach %.0f%% coverage on the %d vitals; "
                 "%d / %d have all %d labs in the first %d h; %d pass both.",
                 len(pass_vitals), n_eligible, c["min_panel_coverage"] * 100, len(vitals),
                 len(pass_labs), n_eligible, len(labs), obs_h, len(keep_stays))
    else:
        n_vars = eligible_stays.attrs.get("n_vars", len(item_mapping))
        expected_bins = total_h * n_vars
        coverage = panel.groupby("stay_id").size() / expected_bins
        keep_stays = coverage[coverage >= c["min_panel_coverage"]].index

    log.info("Stays passing %.0f%% panel-coverage threshold: %d / %d",
              c["min_panel_coverage"] * 100, len(keep_stays), n_eligible)

    # Sorted, so the seeded sample below (by row position) and assign_splits
    # draw the same stays every time. Before, the order was whatever DuckDB's
    # parallel join returned: the same seed drew a different cohort on
    # Nibi than on Trillium (1509 of 3000 stays in common, 2026-10-10).
    cohort = eligible_stays[eligible_stays["stay_id"].isin(keep_stays)].sort_values("stay_id").reset_index(drop=True)
    if c["max_patients"] is not None and len(cohort) > c["max_patients"]:
        cohort = cohort.sample(n=c["max_patients"], random_state=c["random_seed"]).reset_index(drop=True)
        log.info("Subsampled cohort to max_patients=%d for this run.", c["max_patients"])

    panel = panel[panel["stay_id"].isin(cohort["stay_id"])]
    return cohort, panel


def assign_splits(cohort: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cohort"]
#    rng = np.random.RandomState(c["random_seed"])
    ids = cohort["stay_id"].sample(frac=1.0, random_state=c["random_seed"]).tolist()
    n = len(ids)
    n_train = int(n * c["train_frac"])
    n_val = int(n * c["val_frac"])
    split_map = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    cohort = cohort.copy()
    cohort["split"] = cohort["stay_id"].map(split_map)
    log.info("Split sizes: %s", cohort["split"].value_counts().to_dict())
    return cohort


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_work_dirs(cfg)
    require_mimic_layout(cfg)

    mapping_path = Path(cfg["paths"]["cache_dir"]) / "item_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found — run src/resolve_items.py first.")
    item_mapping = json.loads(mapping_path.read_text())

    con = get_duckdb_connection(cfg)
    eligible = build_eligible_stays(con, cfg)
    eligible.attrs["n_vars"] = len(cfg["variables"])

    cache_panel_raw(con, cfg, eligible, item_mapping)
    cache_vasopressor_events(con, cfg, eligible)
    # Release DuckDB's buffer pool before the pandas-heavy resampling below,
    # so the two phases don't hold memory at the same time.
    con.close()

    cohort, panel_long = resample_and_filter(cfg, eligible, item_mapping)
    cohort = assign_splits(cohort, cfg)

    work_dir = Path(cfg["paths"]["work_dir"])
    cohort.to_parquet(work_dir / "cohort.parquet", index=False)
    panel_long.to_parquet(work_dir / "panel_long.parquet", index=False)
    log.info("Cohort extraction complete: %d stays, %d panel rows. Written to %s",
              len(cohort), len(panel_long), work_dir)


if __name__ == "__main__":
    doc= __doc__ or "Build analytic cohort and extract tight variable panel from MIMIC-IV."
    args = parse_step_args(doc.strip().splitlines()[0])
    main(args.config_file)
