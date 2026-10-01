"""
Shared utilities: config loading, path handling, logging.
No cloud, no network calls except to the local Ollama server (127.0.0.1).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"


def parse_step_args(description: str | None = None) -> argparse.Namespace:
    """Common CLI for every step script. `--config` is kept as an alias of
    `--config-file` so existing job scripts and commands keep working."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-file", "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    if not Path(args.config_file).is_file():
        parser.error(f"config file not found: {args.config_file}")
    return args


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Expand ~ and $ENV_VARS (e.g. $PROJECT, $SCRATCH on an HPC cluster) in
    # every path entry. Plain ~-expansion was enough for the original single
    # local-machine setup; the Nibi configs (config_nibi_lean.yaml,
    # config_nibi_full_variables.yaml) rely on $PROJECT/$SCRATCH, which
    # Alliance Canada sets in the job environment, so this needs to run
    # before expanduser() for those to resolve correctly.
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str(Path(os.path.expandvars(val)).expanduser())

    # Backward-compatible default so configs written before the `performance`
    # section was added (e.g. the original 1080 Ti / 32 GB profile) still load.
    cfg.setdefault("performance", {
        "duckdb_threads": os.cpu_count() or 4,
        "duckdb_memory_limit_gb": 8,
        "llm_max_concurrent_requests": 1,
        "batch_predict_baselines": False,
    })
    return cfg


def get_duckdb_connection(cfg: dict):
    """One place that configures every DuckDB connection in this project, so
    the thread/memory budget set in config.yaml's `performance` section is
    honoured everywhere (src/resolve_items.py and src/extract_cohort.py both
    call this instead of duckdb.connect() directly). DuckDB does not use the
    GPU — this only affects the CPU/RAM-bound extraction step — but on a
    24-core / 400GB-RAM machine the defaults DuckDB would pick on its own are
    far below what's actually available, so we set them explicitly rather
    than rely on auto-detection inside a container or restricted cgroup."""
    import duckdb
    perf = cfg.get("performance", {})
    con = duckdb.connect()
    threads = perf.get("duckdb_threads", os.cpu_count() or 4)
    mem_gb = perf.get("duckdb_memory_limit_gb", 8)
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA memory_limit='{int(mem_gb)}GB'")
    # When DuckDB hits memory_limit it spills to temp_directory, which for an
    # in-memory database defaults to ./.tmp — i.e. the (networked) project
    # filesystem when run from the repo. Inside a Slurm job, spill to the
    # node-local $SLURM_TMPDIR instead; it is fast and cleaned up with the job.
    temp_dir = perf.get("duckdb_temp_dir") or os.environ.get("SLURM_TMPDIR")
    if temp_dir:
        temp_dir = str(Path(os.path.expandvars(temp_dir)).expanduser() / "duckdb_tmp")
        con.execute(f"SET temp_directory='{temp_dir}'")
    return con


def ensure_work_dirs(cfg: dict) -> None:
    for key in ("work_dir", "cache_dir", "results_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("TWIN_LOG_LEVEL", "INFO"))
    return logger


def hosp_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["mimic_root"]) / "hosp"


def icu_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["mimic_root"]) / "icu"


def require_mimic_layout(cfg: dict) -> None:
    """Fail fast with a clear message if ~/mimic-iv doesn't look right."""
    h, i = hosp_dir(cfg), icu_dir(cfg)
    missing = [str(p) for p in (h, i) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected MIMIC-IV 'hosp' and 'icu' module folders not found: "
            f"{missing}. This project assumes the raw PhysioNet export is placed at "
            f"{cfg['paths']['mimic_root']}/hosp and {cfg['paths']['mimic_root']}/icu "
            "(the standard MIMIC-IV directory layout, files may be .csv or .csv.gz)."
        )
