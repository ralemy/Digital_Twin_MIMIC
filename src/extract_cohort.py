"""
Step 1: build the analytic cohort and extract the tight variable panel, entirely
locally with DuckDB running directly over the raw PhysioNet .csv/.csv.gz files
under ~/mimic-iv/{hosp,icu}. No data leaves the machine at any point.

Inclusion criteria (Chapter 5, Section 5.3):
  - first ICU stay per patient
  - age >= 18 at admission
  - ICU stay length >= 48 hours
  - >=50% of the tight panel's expected measurements populated in the
    observation + horizon window

Outputs (written under <work_dir>):
  cache/chartevents_panel.parquet   filtered raw chartevents rows for the panel itemids
  cache/labevents_panel.parquet     filtered raw labevents rows (lactate)
  cache/vasopressor_events.parquet  filtered inputevents rows used for RQ3 labels
  cohort.parquet                    one row per eligible stay_id, with split assignment
  panel_long.parquet                long-format (stay_id, variable, hour, value) resampled panel

Usage:
    python src/extract_cohort.py --config config/config.yaml
Requires resolve_items.py to have been run first (needs cache/item_mapping.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from common import load_config, ensure_work_dirs, get_logger, hosp_dir, icu_dir, require_mimic_layout, get_duckdb_connection

log = get_logger("extract_cohort")


def _find_file(directory: Path, stem: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.csv.gz under {directory}")


def build_eligible_stays(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    icustays_path = _find_file(icu_dir(cfg), "icustays")
    patients_path = _find_file(hosp_dir(cfg), "patients")
    admissions_path = _find_file(hosp_dir(cfg), "admissions")

    c = cfg["cohort"]
    first_stay_clause = "AND rn = 1" if c["first_stay_only"] else ""

    query = f"""
        WITH stays AS (
            SELECT
                s.subject_id, s.hadm_id, s.stay_id,
                s.intime, s.outtime,
                date_diff('hour', s.intime, s.outtime) AS los_hours,
                row_number() OVER (PARTITION BY s.subject_id ORDER BY s.intime) AS rn
            FROM read_csv_auto(?, ignore_errors=true) s
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
        WHERE st.los_hours >= {c['min_icu_stay_hours']}
          AND p.anchor_age >= {c['min_age_years']}
          {first_stay_clause}
    """
    df = con.execute(query, [str(icustays_path), str(patients_path), str(admissions_path)]).fetchdf()
    log.info("Stays passing age/LOS/first-stay filters: %d", len(df))
    return df


def cache_panel_raw(con: duckdb.DuckDBPyConnection, cfg: dict, eligible_stays: pd.DataFrame, item_mapping: dict) -> None:
    """One filtered pass over chartevents and labevents; cached to parquet so
    later steps (and re-runs) never re-scan the full raw tables again.

    Filters are pushed down via registered temp tables (semi-joins) rather
    than string-interpolated IN (...) lists, so this stays correct and fast
    even when the eligible cohort or itemid set is large."""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    stay_ids = eligible_stays["stay_id"].tolist()
    hadm_ids = eligible_stays["hadm_id"].unique().tolist()

    con.register("eligible_stays_df", eligible_stays[["stay_id", "hadm_id", "intime"]])
    con.register("stay_ids_df", pd.DataFrame({"stay_id": stay_ids}))
    con.register("hadm_ids_df", pd.DataFrame({"hadm_id": hadm_ids}))

    chart_itemids = []
    lab_itemids = []
    for var, info in item_mapping.items():
        if info["source"] == "icu_chartevents":
            chart_itemids.extend(info["itemids"])
        elif info["source"] == "hosp_labevents":
            lab_itemids.extend(info["itemids"])

    if chart_itemids:
        chartevents_path = _find_file(icu_dir(cfg), "chartevents")
        out_path = cache_dir / "chartevents_panel.parquet"
        if not out_path.exists():
            log.info("Scanning chartevents for %d itemids over %d stays (one-time pass, can take a while)...",
                      len(chart_itemids), len(stay_ids))
            con.register("chart_itemids_df", pd.DataFrame({"itemid": chart_itemids}))
            con.execute(f"""
                COPY (
                    SELECT ce.stay_id, ce.itemid, ce.charttime, ce.valuenum
                    FROM read_csv_auto('{chartevents_path}', ignore_errors=true) ce
                    WHERE ce.stay_id IN (SELECT stay_id FROM stay_ids_df)
                      AND ce.itemid IN (SELECT itemid FROM chart_itemids_df)
                      AND ce.valuenum IS NOT NULL
                ) TO '{out_path}' (FORMAT PARQUET)
            """)
            log.info("Wrote %s", out_path)
        else:
            log.info("Reusing cached %s", out_path)

    if lab_itemids:
        labevents_path = _find_file(hosp_dir(cfg), "labevents")
        out_path = cache_dir / "labevents_panel.parquet"
        if not out_path.exists():
            log.info("Scanning labevents for %d itemids over %d admissions (one-time pass)...",
                      len(lab_itemids), len(hadm_ids))
            con.register("lab_itemids_df", pd.DataFrame({"itemid": lab_itemids}))
            con.execute(f"""
                COPY (
                    SELECT le.hadm_id, le.itemid, le.charttime, le.valuenum
                    FROM read_csv_auto('{labevents_path}', ignore_errors=true) le
                    WHERE le.itemid IN (SELECT itemid FROM lab_itemids_df)
                      AND le.hadm_id IN (SELECT hadm_id FROM hadm_ids_df)
                      AND le.valuenum IS NOT NULL
                ) TO '{out_path}' (FORMAT PARQUET)
            """)
            log.info("Wrote %s", out_path)
        else:
            log.info("Reusing cached %s", out_path)


def cache_vasopressor_events(con: duckdb.DuckDBPyConnection, cfg: dict, eligible_stays: pd.DataFrame) -> None:
    """Best-effort extraction of vasopressor administration events for the RQ3
    deterioration label. Not required for RQ1/RQ2; failures here are logged and
    skipped rather than fatal."""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    out_path = cache_dir / "vasopressor_events.parquet"
    if out_path.exists():
        log.info("Reusing cached %s", out_path)
        return
    try:
        d_items_path = _find_file(icu_dir(cfg), "d_items")
        inputevents_path = _find_file(icu_dir(cfg), "inputevents")
    except FileNotFoundError as e:
        log.warning("Skipping vasopressor extraction (RQ3 subgroup will be unavailable): %s", e)
        return

    labels = cfg["deterioration_labels"]["vasopressor_itemid_labels"]
    like_clauses = " OR ".join(["label ILIKE '%' || ? || '%'" for _ in labels])
    vaso_itemids = con.execute(
        f"SELECT DISTINCT itemid FROM read_csv_auto(?, ignore_errors=true) WHERE {like_clauses}",
        [str(d_items_path)] + labels,
    ).fetchdf()["itemid"].tolist()

    if not vaso_itemids:
        log.warning("No vasopressor itemids resolved; RQ3 subgroup will be unavailable.")
        return

    stay_ids = eligible_stays["stay_id"].tolist()
    con.register("vaso_itemids_df", pd.DataFrame({"itemid": vaso_itemids}))
    con.register("vaso_stay_ids_df", pd.DataFrame({"stay_id": stay_ids}))
    con.execute(f"""
        COPY (
            SELECT stay_id, itemid, starttime
            FROM read_csv_auto('{inputevents_path}', ignore_errors=true)
            WHERE itemid IN (SELECT itemid FROM vaso_itemids_df)
              AND stay_id IN (SELECT stay_id FROM vaso_stay_ids_df)
        ) TO '{out_path}' (FORMAT PARQUET)
    """)
    log.info("Wrote %s", out_path)


def resample_and_filter(cfg: dict, eligible_stays: pd.DataFrame, item_mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample raw events to hourly bins per stay/variable, compute panel
    coverage over the observation+horizon window, and drop stays below the
    coverage threshold. Returns (cohort_df, panel_long_df)."""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    c = cfg["cohort"]
    obs_h, hor_h = c["observation_window_hours"], c["forecast_horizon_hours"]
    total_h = obs_h + hor_h

    frames = []
    chart_path = cache_dir / "chartevents_panel.parquet"
    lab_path = cache_dir / "labevents_panel.parquet"

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
    raw["hour_bin"] = np.floor(raw["hour"]).astype(int)

    panel = (
        raw.groupby(["stay_id", "variable", "hour_bin"])["valuenum"]
        .mean()
        .reset_index()
        .rename(columns={"hour_bin": "hour", "valuenum": "value"})
    )

    n_vars = eligible_stays.attrs.get("n_vars", len(item_mapping))
    expected_bins = total_h * n_vars
    coverage = panel.groupby("stay_id").size() / expected_bins
    keep_stays = coverage[coverage >= c["min_panel_coverage"]].index

    log.info("Stays passing %.0f%% panel-coverage threshold: %d / %d",
              c["min_panel_coverage"] * 100, len(keep_stays), eligible_stays["stay_id"].nunique())

    cohort = eligible_stays[eligible_stays["stay_id"].isin(keep_stays)].copy()
    if c["max_patients"] is not None and len(cohort) > c["max_patients"]:
        cohort = cohort.sample(n=c["max_patients"], random_state=c["random_seed"]).reset_index(drop=True)
        log.info("Subsampled cohort to max_patients=%d for this run.", c["max_patients"])

    panel = panel[panel["stay_id"].isin(cohort["stay_id"])]
    return cohort, panel


def assign_splits(cohort: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cohort"]
    rng = np.random.RandomState(c["random_seed"])
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

    cohort, panel_long = resample_and_filter(cfg, eligible, item_mapping)
    cohort = assign_splits(cohort, cfg)

    work_dir = Path(cfg["paths"]["work_dir"])
    cohort.to_parquet(work_dir / "cohort.parquet", index=False)
    panel_long.to_parquet(work_dir / "panel_long.parquet", index=False)
    log.info("Cohort extraction complete: %d stays, %d panel rows. Written to %s",
              len(cohort), len(panel_long), work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
