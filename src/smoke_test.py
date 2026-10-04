"""
Synthetic end-to-end smoke test — NOT part of the real experiment.

Fabricates a small synthetic cohort/panel in the same shape extract_cohort.py
would produce, and a mock LLM (no Ollama needed), then runs the full
pipeline (harmonization -> similarity -> forecasting -> critic, plus the
naive/GBM/LSTM baselines) end to end so you can confirm the codebase runs
correctly on your machine BEFORE pointing it at real, credentialed MIMIC-IV
data. Takes well under a minute.

Usage:
    python src/smoke_test.py [--config-file config/config.yaml]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import get_logger, load_config, parse_step_args

log = get_logger("smoke_test")


def make_synthetic_config(config_path: str, tmp_dir: Path) -> dict:
    cfg = load_config(config_path)
    for key in ("work_dir", "cache_dir", "results_dir"):
        cfg["paths"][key] = str(tmp_dir / key)
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    cfg["cohort"]["max_patients"] = 40
    cfg["cohort"]["observation_window_hours"] = 24
    cfg["cohort"]["forecast_horizon_hours"] = 12  # shorter horizon for a fast smoke test
    # Exercise the '@variant' path too (the mock LLM ignores the model name).
    cfg["llm"]["variants"] = {"smoke_alt": {"model": "hf.co/smoke/alt-model-GGUF:Q4_K_M", "alias": "smoke-alt"}}
    cfg["conditions"] = [c for c in cfg["conditions"] if "@" not in c] + [
        "single_model_llm@smoke_alt", "full_pipeline@smoke_alt"]
    return cfg


def make_synthetic_cohort_and_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(0)
    n = cfg["cohort"]["max_patients"]
    stay_ids = list(range(1, n + 1))
    cohort = pd.DataFrame({
        "subject_id": stay_ids,
        "hadm_id": stay_ids,
        "stay_id": stay_ids,
        "intime": pd.Timestamp("2150-01-01"),
        "outtime": pd.Timestamp("2150-01-04"),
        "los_hours": 72,
        "age_at_admission": rng.randint(18, 90, size=n),
    })
    n_train = int(n * cfg["cohort"]["train_frac"])
    n_val = int(n * cfg["cohort"]["val_frac"])
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    rng.shuffle(splits)
    cohort["split"] = splits

    variables = [v["name"] for v in cfg["variables"]]
    var_baseline = {"heart_rate": 85, "resp_rate": 18, "spo2": 97, "map": 80, "lactate": 1.5}
    total_h = cfg["cohort"]["observation_window_hours"] + cfg["cohort"]["forecast_horizon_hours"]

    rows = []
    for sid in stay_ids:
        for var in variables:
            base = var_baseline.get(var, 50)
            trend = rng.normal(0, 0.05)
            for h in range(total_h):
                if rng.rand() < 0.15:  # simulate missingness
                    continue
                val = base + trend * h + rng.normal(0, base * 0.05)
                rows.append({"stay_id": sid, "variable": var, "hour": h, "value": val})
    panel_long = pd.DataFrame(rows)
    return cohort, panel_long


def mock_llm_generate(self, prompt, system=None, json_mode=False):
    """Deterministic fake LLM: returns a plausible-looking forecast JSON
    without any real model, so the smoke test doesn't require Ollama."""
    import re
    variables = ["heart_rate", "resp_rate", "spo2", "map", "lactate"]
    horizon = 12
    if "corrected_values" in system if system else False:
        pass
    if "Revise ONLY the implausible" in (system or ""):
        # correction call
        nums = re.findall(r"[-+]?\d*\.?\d+", prompt.split("Hourly forecast values:")[-1])
        vals = [float(x) for x in nums[:horizon]] if nums else [80.0] * horizon
        return json.dumps({"corrected_values": vals})
    forecast = {v: [80.0 + i * 0.1 for i in range(horizon)] for v in variables}
    halfwidth = {v: 5.0 for v in variables}
    return json.dumps({"forecast": forecast, "interval_halfwidth": halfwidth})


def main(config_path: str) -> None:
    # Node-local in a job, the system temp directory elsewhere.
    tmp_dir = Path(os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()) / "mimic_twin_smoke_test"
    cfg = make_synthetic_config(config_path, tmp_dir)
    cohort, panel_long = make_synthetic_cohort_and_panel(cfg)

    work_dir = Path(cfg["paths"]["work_dir"])
    cohort.to_parquet(work_dir / "cohort.parquet", index=False)
    panel_long.to_parquet(work_dir / "panel_long.parquet", index=False)
    log.info("Synthetic cohort (%d stays) and panel (%d rows) written to %s", len(cohort), len(panel_long), work_dir)

    from harmonization_agent import build_tensors
    tensors = build_tensors(cfg, cohort, panel_long)

    from run_experiment import split_tensors
    splits = split_tensors(tensors, cohort)

    with patch("llm_client.LocalLLM._check_server", lambda self: None), \
         patch("llm_client.LocalLLM.generate", mock_llm_generate):
        from pipeline import fit_models, run_condition, validate_conditions
        validate_conditions(cfg)
        fitted = fit_models(cfg, splits["train"])
        alt_llm = fitted["variants"]["smoke_alt"]["forecaster"].llm
        assert alt_llm.model == "smoke-alt", f"variant alias not used: {alt_llm.model}"

        from metrics import evaluate_twin
        variables = [v["name"] for v in cfg["variables"]]

        for condition in cfg["conditions"]:
            result = run_condition(condition, cfg, splits["train"], splits["test"], fitted)
            report = evaluate_twin(result["y_true"], result["y_pred"], variables,
                                    y_lower=result["y_lower"], y_upper=result["y_upper"])
            log.info("[SMOKE TEST] Condition '%s' ran successfully:\n%s\n", condition, report.summary_table())

    log.info("SMOKE TEST PASSED: full pipeline ran end to end with no errors.")
    log.info("Synthetic work dir: %s (safe to delete)", tmp_dir)


if __name__ == "__main__":
    args = parse_step_args("Arguments to smoke test")
    main(args.config_file)
