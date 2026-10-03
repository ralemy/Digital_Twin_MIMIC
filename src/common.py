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
from typing import Callable

import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"


def parse_step_args(
    description: str | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Common CLI for every step script. `--config` is kept as an alias of
    `--config-file` so existing job scripts and commands keep working.
    `add_arguments(parser)` lets a step add its own flags (e.g.
    run_experiment.py's --full-refresh)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-file", "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    if add_arguments is not None:
        add_arguments(parser)
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


LLM_CONDITIONS = ("single_model_llm", "full_pipeline", "full_pipeline_no_critic", "full_pipeline_no_similarity")


def split_condition(condition: str) -> tuple[str, str | None]:
    """A condition may name an alternative LLM after '@', e.g.
    'full_pipeline@medgemma' -> ('full_pipeline', 'medgemma');
    'full_pipeline' -> ('full_pipeline', None), i.e. the default llm.model."""
    base, sep, variant = condition.partition("@")
    return base, (variant if sep else None)


def cfg_for_llm_variant(cfg: dict, variant: str | None) -> dict:
    """The config as seen by one LLM variant: llm.variants.<variant> overlaid
    on the llm section, so a variant only needs to list what differs (usually
    just `model`, sometimes num_ctx/max_tokens). variant=None returns cfg."""
    if variant is None:
        return cfg
    variants = cfg["llm"].get("variants") or {}
    if variant not in variants:
        raise ValueError(f"LLM variant '{variant}' is used in `conditions` but not defined "
                         f"under llm.variants (defined: {sorted(variants)})")
    llm = {k: v for k, v in cfg["llm"].items() if k != "variants"}
    # A variant that swaps the model must not inherit the default model's
    # alias — that alias names a different model in `ollama list`.
    if "model" in variants[variant] and "alias" not in variants[variant]:
        llm.pop("alias", None)
    llm.update(variants[variant])
    return {**cfg, "llm": llm}


def ollama_model_name(llm_cfg: dict) -> str:
    """The name to send to Ollama: llm.alias if set, else llm.model. `model`
    is the tag you pull (often a long hf.co/... path); `alias` is an optional
    short local name for it, created once with `ollama cp <model> <alias>`."""
    return llm_cfg.get("alias") or llm_cfg["model"]


def llm_variants_in_use(cfg: dict) -> list[str | None]:
    """The LLM variants a run will build, None meaning the default llm.model.
    The default is built if run_single_model_llm is set or any full_pipeline*
    condition has no '@variant'; a variant only if a condition names it."""
    parsed = [split_condition(c) for c in cfg["conditions"]]
    out: list[str | None] = []
    if cfg["baselines"]["run_single_model_llm"] or any(
        "full_pipeline" in base and variant is None for base, variant in parsed
    ):
        out.append(None)
    for base, variant in parsed:
        if variant is not None and base in LLM_CONDITIONS and variant not in out:
            out.append(variant)
    return out


def llm_models_in_use(cfg: dict) -> list[str]:
    """Ollama model names (alias where one is set) a run needs present in
    `ollama list`, e.g. for the job script's pre-flight check:
    `python -c '...print(*llm_models_in_use(cfg))'`."""
    models = [ollama_model_name(cfg_for_llm_variant(cfg, v)["llm"]) for v in llm_variants_in_use(cfg)]
    return list(dict.fromkeys(models))


def llm_models_to_set_up(cfg: dict) -> list[tuple[str, str | None]]:
    """(model, alias) for every LLM a run needs, alias None where none is set,
    e.g. for jobs/prep2_download_models.sh."""
    pairs = []
    for v in llm_variants_in_use(cfg):
        llm = cfg_for_llm_variant(cfg, v)["llm"]
        pairs.append((llm["model"], llm.get("alias") or None))
    return list(dict.fromkeys(pairs))


def llm_setup_commands(cfg: dict) -> list[str]:
    """Login-node commands that make every model a run needs available:
    `ollama pull <model>`, plus `ollama cp <model> <alias>` where an alias is set."""
    return [f"ollama pull {model}" + (f" && ollama cp {model} {alias}" if alias else "")
            for model, alias in llm_models_to_set_up(cfg)]


def vasopressor_window_hours(cfg: dict) -> int:
    """Vasopressor events are kept from ICU admission up to the end of the
    RQ3 lookahead window (observation cutoff + lookahead_hours)."""
    return cfg["cohort"]["observation_window_hours"] + cfg["deterioration_labels"]["lookahead_hours"]


def vasopressor_cache_path(cfg: dict) -> Path:
    """Written by extract_cohort.py, read by evaluate_results.py. The window
    is part of the filename so changing it triggers a fresh extraction
    instead of silently reusing a cache built for a different window."""
    return Path(cfg["paths"]["cache_dir"]) / f"vasopressor_events_{vasopressor_window_hours(cfg)}h.parquet"


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
