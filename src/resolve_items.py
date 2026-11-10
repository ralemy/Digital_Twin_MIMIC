"""
Step 0: resolve the tight variable panel's item labels (the config) to concrete
MIMIC-IV itemids, by querying icu/d_items and hosp/d_labitems locally with DuckDB.

Rationale: hardcoding itemids is fragile across MIMIC-IV point releases and easy
to get subtly wrong. Resolving by label pattern against the dataset actually on
disk, then writing the resolved mapping to a cache file for the researcher to
eyeball and correct if needed, is safer and is the same approach documented in
the proposal's Appendix A ("variable-panel item identifier mapping").

Usage:
    python src/resolve_items.py --config-file config/config_local.yaml
Writes:
    <cache_dir>/item_mapping.json
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from common import (
    ensure_work_dirs,
    get_duckdb_connection,
    get_logger,
    hosp_dir,
    icu_dir,
    load_config,
    parse_step_args,
    require_mimic_layout,
)

log = get_logger("resolve_items")


def _find_file(directory: Path, stem: str) -> Path:
    """MIMIC-IV ships tables as either <stem>.csv or <stem>.csv.gz."""
    for ext in (".csv.gz", ".csv"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.csv.gz under {directory}")


def _label_clauses(patterns: list[str], exact: bool) -> str:
    """Case-insensitive label match: the label must contain a pattern, or with
    exact=True be equal to one. Exact matching is for short labels such as
    "pH" or "Hemoglobin", which as substrings match dozens of unrelated items
    (Phosphate, Lymphocytes, Carboxyhemoglobin, ...)."""
    clause = "label ILIKE ?" if exact else "label ILIKE '%' || ? || '%'"
    return " OR ".join([clause for _ in patterns])


def resolve_icu_item(con: duckdb.DuckDBPyConnection, d_items_path: Path, patterns: list[str], excludes: list[str],
                     exact: bool = False) -> list[dict]:
    like_clauses = _label_clauses(patterns, exact)
    query = f"""
        SELECT itemid, label, category, param_type, unitname
        FROM read_csv_auto(?, ignore_errors=true)
        WHERE ({like_clauses})
    """
    params = [str(d_items_path)] + patterns
    rows = con.execute(query, params).fetchall()
    cols = ["itemid", "label", "category", "param_type", "unitname"]
    results = [dict(zip(cols, r)) for r in rows]
    if excludes:
        results = [
            r for r in results
            if not any(ex.lower() in (r["label"] or "").lower() for ex in excludes)
        ]
    return results


def resolve_lab_item(con: duckdb.DuckDBPyConnection, d_labitems_path: Path, patterns: list[str], excludes: list[str],
                     exact: bool = False, fluids: list[str] | None = None) -> list[dict]:
    like_clauses = _label_clauses(patterns, exact)
    query = f"""
        SELECT itemid, label, fluid, category
        FROM read_csv_auto(?, ignore_errors=true)
        WHERE ({like_clauses})
    """
    params = [str(d_labitems_path)] + patterns
    rows = con.execute(query, params).fetchall()
    cols = ["itemid", "label", "fluid", "category"]
    results = [dict(zip(cols, r)) for r in rows]
    if excludes:
        results = [
            r for r in results
            if not any(ex.lower() in (r["label"] or "").lower() for ex in excludes)
        ]
    if fluids:
        # d_labitems has a separate `fluid` column (Blood, Urine, Pleural,
        # Cerebrospinal Fluid, ...); filtering on it is more reliable than
        # excluding fluid names from labels, which many items don't include.
        wanted = {f.lower() for f in fluids}
        results = [r for r in results if (r["fluid"] or "").lower() in wanted]
    return results


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_work_dirs(cfg)
    require_mimic_layout(cfg)

    d_items_path = _find_file(icu_dir(cfg), "d_items")
    d_labitems_path = _find_file(hosp_dir(cfg), "d_labitems")

    con = get_duckdb_connection(cfg)
    mapping = {}

    for var in cfg["variables"]:
        patterns = var["label_patterns"]
        excludes = var.get("exclude_patterns", [])
        match = var.get("match", "contains")
        if match not in ("contains", "exact"):
            raise ValueError(f"Unknown match '{match}' for variable '{var['name']}' (use 'contains' or 'exact')")
        exact = match == "exact"
        if var["source"] == "icu_chartevents":
            if var.get("fluids"):
                raise ValueError(f"'fluids' only applies to hosp_labevents variables (variable '{var['name']}')")
            matches = resolve_icu_item(con, d_items_path, patterns, excludes, exact=exact)
        elif var["source"] == "hosp_labevents":
            matches = resolve_lab_item(con, d_labitems_path, patterns, excludes, exact=exact,
                                       fluids=var.get("fluids"))
        else:
            raise ValueError(f"Unknown source '{var['source']}' for variable '{var['name']}'")

        if not matches:
            log.warning(
                "No itemid matched for variable '%s' (patterns=%s). "
                "You will need to inspect %s manually and edit the cache file.",
                var["name"], patterns, d_items_path if var["source"] == "icu_chartevents" else d_labitems_path,
            )
        else:
            log.info("Resolved '%s' -> %d itemid(s): %s", var["name"], len(matches), [m["itemid"] for m in matches])

        mapping[var["name"]] = {
            "source": var["source"],
            "itemids": [m["itemid"] for m in matches],
            "matches": matches,  # kept for human review
        }

    out_path = Path(cfg["paths"]["cache_dir"]) / "item_mapping.json"
    out_path.write_text(json.dumps(mapping, indent=2))
    log.info("Wrote item mapping to %s — REVIEW THIS FILE before running extraction.", out_path)


if __name__ == "__main__":
    doc = __doc__ or "Resolve variable panel item labels to MIMIC-IV itemids."
    args = parse_step_args(doc.strip().splitlines()[0])
    main(args.config_file)
