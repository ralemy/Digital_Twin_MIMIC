"""
Optional live metrics on Weights & Biases (logging.wandb in the config).

Privacy rule: only aggregate numbers (progress, rates, per-condition
metrics, tuning scores) and fixed error-type messages are sent — never log
lines, model output, per-patient values, stay ids or files. W&B's console
capture, code and git snapshots are turned off. Errors are reported only as
a count per type plus a fixed message such as "malformed forecast - see logs
for details". W&B also records system metrics (CPU, memory, GPU) of the
job's node, which covers the Ollama server's GPU use.

Never breaks a run: if wandb is missing, disabled, not logged in, or the
service is unreachable, every call here is a no-op (one warning).

Off unless both the config enables it and the profile allows it: the
profile's environment.disable_wandb (default true) is exported by
jobs/setup_bash.sh as DT_WANDB_DISABLED=1 and wins over the config.

Config:
    logging:
      wandb:
        enabled: true
        project: mimic-iv-digital-twin
        entity: null          # your default W&B entity
        mode: online          # or offline (sync later with `wandb sync`)

Setup once, on a login node: `wandb login` (stores the API key in ~/.netrc).
"""
from __future__ import annotations

import os
from pathlib import Path

from common import get_logger

log = get_logger("tracking")

# The only error texts ever sent: the type, never the details.
ERROR_MESSAGES = {
    "malformed_forecast": "malformed forecast - see logs for details",
    "truncated_forecast": "truncated forecast (max_tokens) - see logs for details",
    "llm_server_error": "LLM server error - see logs for details",
    "llm_timeout": "LLM request timed out - see logs for details",
    "critic_correction_failed": "critic correction failed - see logs for details",
    "absurd_values": "forecast held non-finite or absurd values (filled) - see logs for details",
    "context_overflow": "prompt + answer filled num_ctx (context shifted) - see logs for details",
}

_run = None


def start(cfg: dict, stage: str, config_path: str, extra: dict | None = None) -> None:
    """Start a W&B run for one job of a pipeline stage (run, tune, calibrate,
    evaluate). Named <config stem>-<stage>-<Slurm job id>, grouped by config
    and stage, so the chained jobs of one stage sit together."""
    global _run
    wcfg = ((cfg.get("logging") or {}).get("wandb") or {})
    if not wcfg.get("enabled"):
        return
    if os.environ.get("DT_WANDB_DISABLED", "1") == "1":
        log.info("Weights & Biases disabled by the profile (environment.disable_wandb).")
        return
    os.environ["WANDB_CONSOLE"] = "off"            # never upload stdout/stderr
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", os.environ.get("SLURM_TMPDIR") or str(Path(cfg["paths"]["results_dir"])))
    try:
        import wandb
        scope = Path(config_path).stem
        job = os.environ.get("SLURM_JOB_ID", "local")
        settings = wandb.Settings(console="off", save_code=False, disable_code=True, disable_git=True,
                                  x_save_requirements=False)
        _run = wandb.init(
            project=wcfg.get("project", "mimic-iv-digital-twin"), entity=wcfg.get("entity"),
            mode=wcfg.get("mode", "online"), name=f"{scope}-{stage}-{job}", group=f"{scope}-{stage}",
            job_type=stage, settings=settings,
            config={"stage": stage, "config": scope, "slurm_job_id": job,
                    "llm_model": cfg["llm"].get("alias") or cfg["llm"]["model"],
                    "conditions": list(cfg.get("conditions", [])),
                    "n_variables": len(cfg["variables"]), **(extra or {})},
        )
        log.info("Weights & Biases run %s started (aggregate metrics only).", _run.name)
    except Exception as e:  # noqa: BLE001 - tracking must never stop the experiment
        log.warning("Weights & Biases tracking disabled for this job (%s).", type(e).__name__)
        _run = None


def log_metrics(metrics: dict) -> None:
    """Log numbers (non-numeric values are dropped, except ERROR_MESSAGES)."""
    if _run is None:
        return
    safe = {k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) or v in ERROR_MESSAGES.values()}
    try:
        _run.log(safe)
    except Exception as e:  # noqa: BLE001
        log.warning("Weights & Biases log failed (%s); continuing.", type(e).__name__)


def log_errors(prefix: str, counts: dict[str, int]) -> None:
    """Error counts by type (e.g. from ForecastingAgent.error_counts), plus
    the fixed message of each type that occurred."""
    if _run is None or not any(counts.values()):
        return
    metrics = {}
    for kind, n in counts.items():
        if n and kind in ERROR_MESSAGES:
            metrics[f"{prefix}/errors/{kind}"] = n
            metrics[f"{prefix}/errors/{kind}_message"] = ERROR_MESSAGES[kind]
    log_metrics(metrics)


def progress_logger(prefix: str = ""):
    """A run_condition(progress=...) callback: per batch, the condition's
    batches done, fraction done, seconds per batch, fallbacks and error
    counts so far, under <prefix><condition>/... ."""
    def _log(p: dict) -> None:
        key = f"{prefix}{p['condition']}"
        log_metrics({f"{key}/batches_done": p["batch"], f"{key}/fraction_done": p["batch"] / p["n_batches"],
                     f"{key}/sec_per_batch": p["sec_per_batch"], f"{key}/fallbacks": p["fallbacks"]})
        log_errors(key, p.get("errors") or {})
    return _log if _run is not None else None


def summary(values: dict) -> None:
    """Final numbers shown in the run's summary table."""
    if _run is None:
        return
    try:
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                _run.summary[k] = v
    except Exception as e:  # noqa: BLE001
        log.warning("Weights & Biases summary failed (%s); continuing.", type(e).__name__)


def finish(exit_code: int = 0) -> None:
    global _run
    if _run is None:
        return
    try:
        _run.finish(exit_code=exit_code)
    except Exception:  # noqa: BLE001
        pass
    _run = None
