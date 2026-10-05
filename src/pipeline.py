"""
Orchestration — runs every condition in the config's `conditions` list over
the test split and returns stacked (n_patients, horizon_hours, n_variables)
arrays ready for src/metrics.py. This is the "local, script-based sequential
pipeline" described in Chapter 5, Section 5.5, step 2 — a single Python
process, no message queue, no separate services, just function calls in
sequence.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from baselines import GBMBaseline, LSTMBaseline, naive_forecast
from checkpoint import ConditionCheckpoint, RunCheckpoint, model_fingerprint
from common import LLM_CONDITIONS, cfg_for_llm_variant, critic_variant, get_logger, llm_variants_in_use, split_condition
from critic_agent import CriticAgent
from forecasting_agent import ForecastingAgent
from llm_client import LocalLLM
from similarity_agent import SimilarityAgent

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
    checkpoint: ConditionCheckpoint | None = None,
    progress=None,
) -> dict:
    """
    Returns {"y_true": ..., "y_pred": ..., "y_lower": ..., "y_upper": ...,
             "plausibility_violations_precritic": int or None,
             "llm_fallbacks": int or None,
             "llm_filled": per-variable int array or None}
    the first four shaped (n_patients, horizon_hours, n_variables).
    llm_filled counts, per variable, the forecasts that left the variable out
    and got the naive forecast for it (see ForecastingAgent._fill_missing).
    "llm_errors" counts the condition's failed LLM calls by type
    (tracking.ERROR_MESSAGES keys), in the batches computed by this call.

    `progress`, if given, is called after each computed batch with a dict of
    aggregate numbers (batches done, seconds per batch, fallbacks and errors
    so far): the live metrics hook (src/tracking.py).

    An LLM condition may name a model variant after '@' (e.g.
    'full_pipeline@medgemma'); it then runs with that variant's forecaster
    and critic, built by fit_models() from llm.variants.<name>.

    Test patients are processed in batches of performance.checkpoint_batch_size
    (sorted by stay_id). With `checkpoint`, each batch is saved as soon as it
    finishes, and batches saved by an earlier, interrupted run are loaded
    instead of recomputed — so a stopped run loses at most one batch.
    """
    base, _ = split_condition(condition)
    stay_ids = sorted(test_tensors.keys())
    n_workers = max(1, int(cfg.get("performance", {}).get("llm_max_concurrent_requests", 1)))
    batch_size = max(1, int(cfg.get("performance", {}).get("checkpoint_batch_size", 32)))
    n_batches = -(-len(stay_ids) // batch_size)

    y_true = np.stack([test_tensors[sid]["horizon"] for sid in stay_ids])
    y_pred = np.full_like(y_true, np.nan)
    y_lower = np.full_like(y_true, np.nan)
    y_upper = np.full_like(y_true, np.nan)
    total_precritic_violations = 0
    total_fallbacks = 0
    total_filled = np.zeros(y_true.shape[2], dtype=int)
    n_resumed = 0
    agents = (fitted_models["variants"][split_condition(condition)[1]] if split_condition(condition)[1]
              else fitted_models) if base in LLM_CONDITIONS else {}
    errors_before = {name: dict(agent.error_counts) for name, agent in agents.items()
                     if name in ("forecaster", "critic")}

    def errors_so_far() -> dict[str, int]:
        out: dict[str, int] = {}
        for name, before in errors_before.items():
            for kind, n in agents[name].error_counts.items():
                out[kind] = out.get(kind, 0) + n - before.get(kind, 0)
        return {k: v for k, v in out.items() if v}

    log.info("Running condition '%s' over %d test patients in %d batches of up to %d "
             "(llm_max_concurrent_requests=%d)...",
             condition, len(stay_ids), n_batches, batch_size, n_workers)
    t_start = time.time()

    models_loaded = False
    for b, start in enumerate(range(0, len(stay_ids), batch_size)):
        batch_ids = stay_ids[start:start + batch_size]
        batch = checkpoint.load_batch(b, batch_ids) if checkpoint is not None else None
        if batch is not None:
            n_resumed += 1
        else:
            if not models_loaded:
                # Before the first batch computed here: load the condition's
                # models, so its first requests don't time out while Ollama
                # loads them (see llm_client.LOAD_TIMEOUT_S).
                _load_models(base, agents)
                models_loaded = True
            batch = _predict_batch(condition, cfg, batch_ids, test_tensors, fitted_models, n_workers)
            if checkpoint is not None:
                checkpoint.save_batch(b, batch_ids, batch)
            if base in LLM_CONDITIONS:
                done = b + 1 - n_resumed
                elapsed = time.time() - t_start
                log.info("Condition '%s': batch %d/%d done (%.0fs/batch, ~%.0f min left).",
                         condition, b + 1, n_batches, elapsed / done,
                         elapsed / done * (n_batches - b - 1) / 60)
            if progress is not None:
                progress({"condition": condition, "batch": b + 1, "n_batches": n_batches,
                          "sec_per_batch": (time.time() - t_start) / max(1, b + 1 - n_resumed),
                          "fallbacks": total_fallbacks + batch["fallbacks"],
                          "errors": errors_so_far()})

        rows = slice(start, start + len(batch_ids))
        y_pred[rows], y_lower[rows], y_upper[rows] = batch["y_pred"], batch["y_lower"], batch["y_upper"]
        total_precritic_violations += batch["violations"]
        total_fallbacks += batch["fallbacks"]
        total_filled += batch["filled"]

    if n_resumed:
        log.info("Condition '%s': %d of %d batches loaded from checkpoint, %d computed.",
                 condition, n_resumed, n_batches, n_batches - n_resumed)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "stay_ids": stay_ids,
        "plausibility_violations_precritic": total_precritic_violations if "full_pipeline" in base else None,
        "llm_fallbacks": total_fallbacks if base in LLM_CONDITIONS else None,
        "llm_filled": total_filled if base in LLM_CONDITIONS else None,
        "llm_errors": errors_so_far(),
    }


# The conditions whose critic calls its LLM (no_critic skips the critic,
# clip_critic only clips; see _predict_batch).
LLM_CRITIC_CONDITIONS = ("full_pipeline", "full_pipeline_no_similarity")


def _load_models(base: str, agents: dict) -> None:
    """Load the forecaster's model and, if the condition calls the critic's
    LLM and it is another model, the critic's."""
    seen = set()
    for name in ("forecaster", "critic") if base in LLM_CRITIC_CONDITIONS else ("forecaster",):
        llm = getattr(agents.get(name), "llm", None)
        if llm is not None and llm.model not in seen:
            seen.add(llm.model)
            llm.load()


def _predict_batch(
    condition: str,
    cfg: dict,
    batch_ids: list[int],
    test_tensors: dict[int, dict],
    fitted_models: dict,
    n_workers: int,
) -> dict:
    """Predictions for one batch of test patients: {"y_pred", "y_lower",
    "y_upper"} shaped (len(batch_ids), horizon_hours, n_variables), plus this
    batch's pre-critic plausibility "violations", LLM "fallbacks" and
    per-variable "filled" counts."""
    base, variant = split_condition(condition)
    llm_agents = fitted_models["variants"][variant] if variant else fitted_models
    variables = [v["name"] for v in cfg["variables"]]
    fallbacks_before = llm_agents["forecaster"].n_fallbacks if base in LLM_CONDITIONS else 0
    filled_before = dict(llm_agents["forecaster"].n_filled) if base in LLM_CONDITIONS else {}
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]

    shape = (len(batch_ids), horizon_hours, len(variables))
    y_pred = np.full(shape, np.nan)
    y_lower = np.full(shape, np.nan)
    y_upper = np.full(shape, np.nan)
    total_precritic_violations = 0

    if base == "naive":
        for pi, sid in enumerate(batch_ids):
            r = naive_forecast(test_tensors[sid]["obs"], horizon_hours)
            y_pred[pi] = r["forecast_array"]
            y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
            y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif base == "gbm":
        # sklearn's HistGradientBoostingRegressor.predict is CPU-vectorized
        # internally; looping per patient is already fast enough at lean-scope
        # cohort sizes that batching here wouldn't meaningfully help.
        gbm_model: GBMBaseline = fitted_models["gbm"]
        for pi, sid in enumerate(batch_ids):
            r = gbm_model.predict(test_tensors[sid]["obs"])
            y_pred[pi] = r["forecast_array"]
            y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
            y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif base == "lstm":
        lstm_model: LSTMBaseline = fitted_models["lstm"]
        if cfg.get("performance", {}).get("batch_predict_baselines", False) and hasattr(lstm_model, "predict_batch"):
            # One batched forward pass on the GPU instead of one Python-level
            # call per patient — this is where a 40GB card actually pays off
            # for the LSTM baseline at larger cohort sizes; a single-patient
            # forward pass barely uses the GPU at all.
            obs_stack = np.stack([test_tensors[sid]["obs"] for sid in batch_ids])
            batch = lstm_model.predict_batch(obs_stack)
            y_pred[:] = batch["forecast_array"]
            y_lower[:] = batch["forecast_array"] - batch["halfwidth_array"][:, None, :]
            y_upper[:] = batch["forecast_array"] + batch["halfwidth_array"][:, None, :]
        else:
            for pi, sid in enumerate(batch_ids):
                r = lstm_model.predict(test_tensors[sid]["obs"])
                y_pred[pi] = r["forecast_array"]
                y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
                y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif base == "single_model_llm":
        # DT-GPT-style: same LLM as the pipeline, no similarity conditioning, no critic.
        forecaster: ForecastingAgent = llm_agents["forecaster"]

        def _single_model_worker(sid):
            raw = forecaster.forecast(test_tensors[sid]["obs"], similarity_context=None)
            return _llm_forecast_to_arrays(raw, variables, horizon_hours)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Each Ollama HTTP call releases the GIL while waiting on I/O, so a
            # thread pool (not multiprocessing) is enough to get concurrent
            # requests in flight — see performance.llm_max_concurrent_requests
            # in the config and the matching OLLAMA_NUM_PARALLEL note in the
            # README. Results are collected via map(), which preserves order,
            # so y_pred[pi] still lines up with batch_ids[pi].
            for pi, (fcast, interval, _) in enumerate(ex.map(_single_model_worker, batch_ids)):
                y_pred[pi] = fcast
                y_lower[pi] = fcast - interval
                y_upper[pi] = fcast + interval

    elif base in ("full_pipeline", "full_pipeline_no_critic", "full_pipeline_no_similarity",
                  "full_pipeline_clip_critic"):
        forecaster: ForecastingAgent = llm_agents["forecaster"]
        critic: CriticAgent = llm_agents["critic"]
        if base == "full_pipeline_clip_critic":
            critic = CriticAgent(None, cfg, enabled=True, clip_only=True)
        similarity: SimilarityAgent = fitted_models["similarity"]   # model-independent, shared by all variants
        use_similarity = base != "full_pipeline_no_similarity"
        use_critic = base != "full_pipeline_no_critic"

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
            for pi, (fcast, interval, violations) in enumerate(ex.map(_full_pipeline_worker, batch_ids)):
                y_pred[pi] = fcast
                y_lower[pi] = fcast - interval
                y_upper[pi] = fcast + interval
                total_precritic_violations += violations

    else:
        raise ValueError(f"Unknown condition '{condition}'")

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "violations": int(total_precritic_violations),
        "fallbacks": int(llm_agents["forecaster"].n_fallbacks - fallbacks_before) if base in LLM_CONDITIONS else 0,
        "filled": np.array([llm_agents["forecaster"].n_filled[v] - filled_before[v] for v in variables]
                           if base in LLM_CONDITIONS else np.zeros(len(variables)), dtype=int),
    }


def validate_conditions(cfg: dict) -> None:
    """Fail at startup, not hours into a run, on a misspelled condition or an
    undefined / misplaced LLM variant."""
    known = ("naive", "gbm", "lstm") + LLM_CONDITIONS
    for condition in cfg["conditions"]:
        base, variant = split_condition(condition)
        if base not in known:
            raise ValueError(f"Unknown condition '{condition}' (known: {', '.join(known)})")
        if variant is not None:
            if base not in LLM_CONDITIONS:
                raise ValueError(f"Condition '{condition}': only LLM conditions take an '@variant'")
            cfg_for_llm_variant(cfg, variant)   # raises if the variant isn't defined
        cv = critic_variant(cfg, variant) if base in LLM_CONDITIONS else None
        if cv is not None:
            cfg_for_llm_variant(cfg, cv)        # raises if the critic's variant isn't defined
            if critic_variant(cfg, cv) is not None:
                raise ValueError(f"Condition '{condition}': critic variant '{cv}' sets its own "
                                 "critic_variant; a critic's model must be a plain variant")


def fit_models(cfg: dict, train_tensors: dict[int, dict], checkpoint: RunCheckpoint | None = None) -> dict:
    """Fit every model that needs training/indexing once, shared across
    conditions (the LLM-based agents don't need "fitting" beyond loading).
    With `checkpoint`, the GBM and LSTM save their progress there and resume
    from it; the similarity index is cheap and simply refitted."""
    variables = [v["name"] for v in cfg["variables"]]
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
    fitted = {}

    if cfg["baselines"]["run_gbm"]:
        gbm_path = checkpoint.model_path("gbm", model_fingerprint(cfg, "gbm", train_tensors), "gbm.pkl") if checkpoint else None
        b = cfg["baselines"]
        fitted["gbm"] = GBMBaseline(variables, horizon_hours, max_depth=b.get("gbm_max_depth", 4),
                                    max_iter=b.get("gbm_max_iter", 150),
                                    learning_rate=b.get("gbm_learning_rate", 0.1),
                                    ).fit(train_tensors, checkpoint_path=gbm_path)

    if cfg["baselines"]["run_lstm"]:
        lstm_hidden = cfg["baselines"].get("lstm_hidden_size", 64)
        lstm_epochs = cfg["baselines"].get("lstm_epochs", 100)
        lstm_path = checkpoint.model_path("lstm", model_fingerprint(cfg, "lstm", train_tensors), "lstm.pt") if checkpoint else None
        fitted["lstm"] = LSTMBaseline(variables, horizon_hours, hidden=lstm_hidden).fit(
            train_tensors, epochs=lstm_epochs, lr=cfg["baselines"].get("lstm_learning_rate", 1e-3),
            checkpoint_path=lstm_path,
            checkpoint_every=cfg["baselines"].get("lstm_checkpoint_every_epochs", 20),
            seed=cfg["baselines"].get("lstm_seed"),
        )

    in_use = llm_variants_in_use(cfg)
    needs_default_llm = None in in_use
    used_variants = [v for v in in_use if v is not None]

    if needs_default_llm:
        fitted.update(_build_llm_agents(cfg, None))
    fitted["variants"] = {}
    for variant in used_variants:
        fitted["variants"][variant] = _build_llm_agents(cfg, variant)
        llm = fitted["variants"][variant]["forecaster"].llm
        log.info("LLM variant '%s' -> model %s (pulled as %s)", variant, llm.model, llm.source_model)
    if needs_default_llm or used_variants:
        fitted["similarity"] = SimilarityAgent(variables, k=cfg["similarity_agent"]["k_neighbors"]).fit(train_tensors)

    return fitted


def _build_llm_agents(cfg: dict, variant: str | None) -> dict:
    """Forecaster and critic for one LLM variant (None = llm.model). They
    share one LocalLLM, unless the variant sets `critic_variant`: then the
    critic gets its own, built from that variant's llm settings."""
    vcfg = cfg_for_llm_variant(cfg, variant)
    llm = LocalLLM(vcfg)
    cv = critic_variant(cfg, variant)
    critic_llm = LocalLLM(cfg_for_llm_variant(cfg, cv)) if cv is not None else llm
    if cv is not None:
        log.info("LLM variant '%s': critic uses variant '%s' (model %s).", variant, cv, critic_llm.model)
    return {
        "forecaster": ForecastingAgent(llm, vcfg),
        "critic": CriticAgent(critic_llm, vcfg, enabled=vcfg["critic_agent"]["enabled"]),
    }
