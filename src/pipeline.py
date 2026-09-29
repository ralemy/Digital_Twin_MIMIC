"""
Orchestration — runs every condition in config.yaml's `conditions` list over
the test split and returns stacked (n_patients, horizon_hours, n_variables)
arrays ready for src/metrics.py. This is the "local, script-based sequential
pipeline" described in Chapter 5, Section 5.5, step 2 — a single Python
process, no message queue, no separate services, just function calls in
sequence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import get_logger
from baselines import naive_forecast, GBMBaseline, LSTMBaseline
from similarity_agent import SimilarityAgent
from forecasting_agent import ForecastingAgent
from critic_agent import CriticAgent
from llm_client import LocalLLM

log = get_logger("pipeline")


def _llm_forecast_to_arrays(result: dict, variables: list[str], horizon_hours: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast = np.full((horizon_hours, len(variables)), np.nan)
    halfwidth = np.zeros(len(variables))
    for i, var in enumerate(variables):
        vals = result["forecast"].get(var)
        if vals is not None and len(vals) == horizon_hours:
            forecast[:, i] = vals
        halfwidth[i] = result.get("interval_halfwidth", {}).get(var, 1.0)
    return forecast, halfwidth[None, :].repeat(horizon_hours, axis=0), halfwidth


def run_condition(
    condition: str,
    cfg: dict,
    train_tensors: dict[int, dict],
    test_tensors: dict[int, dict],
    fitted_models: dict,
) -> dict:
    """
    Returns {"y_true": ..., "y_pred": ..., "y_lower": ..., "y_upper": ...,
             "plausibility_violations_precritic": int or None}
    all shaped (n_patients, horizon_hours, n_variables) except the last, a scalar.
    """
    variables = [v["name"] for v in cfg["variables"]]
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
    stay_ids = sorted(test_tensors.keys())
    n_workers = max(1, int(cfg.get("performance", {}).get("llm_max_concurrent_requests", 1)))

    y_true = np.stack([test_tensors[sid]["horizon"] for sid in stay_ids])
    y_pred = np.full_like(y_true, np.nan)
    y_lower = np.full_like(y_true, np.nan)
    y_upper = np.full_like(y_true, np.nan)
    total_precritic_violations = 0

    log.info("Running condition '%s' over %d test patients (llm_max_concurrent_requests=%d)...",
              condition, len(stay_ids), n_workers)

    if condition == "naive":
        for pi, sid in enumerate(stay_ids):
            r = naive_forecast(test_tensors[sid]["obs"], horizon_hours)
            y_pred[pi] = r["forecast_array"]
            y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
            y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif condition == "gbm":
        # sklearn's HistGradientBoostingRegressor.predict is CPU-vectorized
        # internally; looping per patient is already fast enough at lean-scope
        # cohort sizes that batching here wouldn't meaningfully help.
        model: GBMBaseline = fitted_models["gbm"]
        for pi, sid in enumerate(stay_ids):
            r = model.predict(test_tensors[sid]["obs"])
            y_pred[pi] = r["forecast_array"]
            y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
            y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif condition == "lstm":
        model: LSTMBaseline = fitted_models["lstm"]
        if cfg.get("performance", {}).get("batch_predict_baselines", False) and hasattr(model, "predict_batch"):
            # One batched forward pass on the GPU instead of one Python-level
            # call per patient — this is where a 40GB card actually pays off
            # for the LSTM baseline at larger cohort sizes; a single-patient
            # forward pass barely uses the GPU at all.
            obs_stack = np.stack([test_tensors[sid]["obs"] for sid in stay_ids])
            batch = model.predict_batch(obs_stack)
            y_pred[:] = batch["forecast_array"]
            y_lower[:] = batch["forecast_array"] - batch["halfwidth_array"][:, None, :]
            y_upper[:] = batch["forecast_array"] + batch["halfwidth_array"][:, None, :]
        else:
            for pi, sid in enumerate(stay_ids):
                r = model.predict(test_tensors[sid]["obs"])
                y_pred[pi] = r["forecast_array"]
                y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
                y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif condition == "single_model_llm":
        # DT-GPT-style: same LLM as the pipeline, no similarity conditioning, no critic.
        forecaster: ForecastingAgent = fitted_models["forecaster"]

        def _single_model_worker(sid):
            raw = forecaster.forecast(test_tensors[sid]["obs"], similarity_context=None)
            return _llm_forecast_to_arrays(raw, variables, horizon_hours)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Each Ollama HTTP call releases the GIL while waiting on I/O, so a
            # thread pool (not multiprocessing) is enough to get concurrent
            # requests in flight — see performance.llm_max_concurrent_requests
            # in config.yaml and the matching OLLAMA_NUM_PARALLEL note in the
            # README. Results are collected via map(), which preserves order,
            # so y_pred[pi] still lines up with stay_ids[pi].
            for pi, (fcast, interval, _) in enumerate(ex.map(_single_model_worker, stay_ids)):
                y_pred[pi] = fcast
                y_lower[pi] = fcast - interval
                y_upper[pi] = fcast + interval

    elif condition in ("full_pipeline", "full_pipeline_no_critic", "full_pipeline_no_similarity"):
        forecaster: ForecastingAgent = fitted_models["forecaster"]
        critic: CriticAgent = fitted_models["critic"]
        similarity: SimilarityAgent = fitted_models["similarity"]
        use_similarity = condition != "full_pipeline_no_similarity"
        use_critic = condition != "full_pipeline_no_critic"

        def _full_pipeline_worker(sid):
            sim_ctx = similarity.query(test_tensors[sid]["obs"]) if use_similarity else None
            raw = forecaster.forecast(test_tensors[sid]["obs"], similarity_context=sim_ctx)

            if use_critic:
                reviewed = critic.review(raw)
                final_forecast = {"forecast": reviewed["forecast"], "interval_halfwidth": raw["interval_halfwidth"]}
                violations = reviewed["violations_before_correction"]
            else:
                final_forecast = raw
                violations = 0

            fcast, interval, _ = _llm_forecast_to_arrays(final_forecast, variables, horizon_hours)
            return fcast, interval, violations

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for pi, (fcast, interval, violations) in enumerate(ex.map(_full_pipeline_worker, stay_ids)):
                y_pred[pi] = fcast
                y_lower[pi] = fcast - interval
                y_upper[pi] = fcast + interval
                total_precritic_violations += violations

    else:
        raise ValueError(f"Unknown condition '{condition}'")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "stay_ids": stay_ids,
        "plausibility_violations_precritic": total_precritic_violations if "full_pipeline" in condition else None,
    }


def fit_models(cfg: dict, train_tensors: dict[int, dict]) -> dict:
    """Fit every model that needs training/indexing once, shared across
    conditions (the LLM-based agents don't need "fitting" beyond loading)."""
    variables = [v["name"] for v in cfg["variables"]]
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
    fitted = {}

    if cfg["baselines"]["run_gbm"]:
        fitted["gbm"] = GBMBaseline(variables, horizon_hours).fit(train_tensors)

    if cfg["baselines"]["run_lstm"]:
        lstm_hidden = cfg["baselines"].get("lstm_hidden_size", 64)
        lstm_epochs = cfg["baselines"].get("lstm_epochs", 100)
        fitted["lstm"] = LSTMBaseline(variables, horizon_hours, hidden=lstm_hidden).fit(
            train_tensors, epochs=lstm_epochs
        )

    needs_llm = (
        cfg["baselines"]["run_single_model_llm"]
        or any("full_pipeline" in c for c in cfg["conditions"])
    )
    if needs_llm:
        llm = LocalLLM(cfg)
        fitted["forecaster"] = ForecastingAgent(llm, cfg)
        fitted["critic"] = CriticAgent(llm, cfg, enabled=cfg["critic_agent"]["enabled"])
        fitted["similarity"] = SimilarityAgent(variables, k=cfg["similarity_agent"]["k_neighbors"]).fit(train_tensors)

    return fitted
