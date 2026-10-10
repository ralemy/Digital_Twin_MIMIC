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

from baselines import PERSISTENCE_BASELINES, GBMBaseline, LSTMBaseline, naive_forecast, persistence_forecast
from checkpoint import ConditionCheckpoint, RunCheckpoint, model_fingerprint
from common import (LLM_CONDITIONS, cfg_for_llm_variant, cfg_for_prompt_group, critic_variant, get_logger,
                    hard_ranges, llm_variants_in_use, prompt_group, split_condition, valid_ranges,
                    validate_per_condition)
from critic_agent import CriticAgent, critic_review_stage
from forecasting_agent import ForecastingAgent, postprocess_forecast, postprocess_params
from harmonization_agent import naive_fill, reference_stats
from llm_client import LocalLLM
from similarity_agent import SimilarityAgent

log = get_logger("pipeline")


def _llm_forecast_to_arrays(result: dict, variables: list[str], horizon_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """(forecast, half-width), both (horizon_hours, n_variables). A
    variable's half-width is one number or one per hour
    (forecasting_agent.interval)."""
    forecast = np.full((horizon_hours, len(variables)), np.nan)
    halfwidth = np.ones((horizon_hours, len(variables)))
    for i, var in enumerate(variables):
        vals = result["forecast"].get(var)
        if vals is not None and len(vals) == horizon_hours:
            forecast[:, i] = vals
        hw = result.get("interval_halfwidth", {}).get(var, 1.0)
        if np.ndim(hw) == 0 or len(hw) == horizon_hours:
            halfwidth[:, i] = hw
    return forecast, halfwidth


def _clip_to_valid(forecast: np.ndarray, cfg: dict) -> np.ndarray:
    """A GBM or LSTM forecast clipped to each variable's valid range: a
    per-hour regressor can otherwise output far outside it (MAP -2759 to
    5395 in the Trillium lean run, fitted on artefact targets)."""
    ranges = valid_ranges(cfg)
    out = forecast.copy()
    for i, var in enumerate(v["name"] for v in cfg["variables"]):
        out[..., i] = np.clip(out[..., i], *ranges[var])
    return out


def run_condition(
    condition: str,
    cfg: dict,
    train_tensors: dict[int, dict],
    test_tensors: dict[int, dict],
    fitted_models: dict,
    checkpoint: ConditionCheckpoint | None = None,
    progress=None,
    postprocess: dict | None = None,
) -> dict:
    """
    `postprocess`: an LLM condition's post-processing parameters
    (forecasting_agent.postprocess_params; default: the config's). Its
    forecasts before post-processing are returned too, as "y_raw", so
    calibrate.py can fit the parameters per condition.

    Returns {"y_true": ..., "y_pred": ..., "y_lower": ..., "y_upper": ...,
             "plausibility_violations_precritic": int or None,
             "critic_clipped": int or None (of those, the values the critic
                               clipped instead of the LLM correcting them),
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
    y_raw = np.full_like(y_true, np.nan)
    params = postprocess if postprocess is not None else postprocess_params(cfg)
    total_precritic_violations = 0
    total_clipped = 0
    total_fallbacks = 0
    total_filled = np.zeros(y_true.shape[2], dtype=int)
    n_resumed = 0
    agents = condition_agents(fitted_models, condition) if base in LLM_CONDITIONS else {}
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
                _load_models(agents)
                models_loaded = True
            batch = _predict_batch(condition, cfg, batch_ids, test_tensors, fitted_models, n_workers, params)
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
        if batch.get("y_raw") is not None:
            y_raw[rows] = batch["y_raw"]
        total_precritic_violations += batch["violations"]
        total_clipped += batch["clipped"]
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
        "y_raw": y_raw if base in LLM_CONDITIONS else None,
        "stay_ids": stay_ids,
        "plausibility_violations_precritic": total_precritic_violations if "full_pipeline" in base else None,
        "critic_clipped": total_clipped if "full_pipeline" in base else None,
        "llm_fallbacks": total_fallbacks if base in LLM_CONDITIONS else None,
        "llm_filled": total_filled if base in LLM_CONDITIONS else None,
        "llm_errors": errors_so_far(),
    }


def condition_agents(fitted_models: dict, condition: str) -> dict:
    """The forecaster and critic an LLM condition runs with: its variant's,
    with the forecaster built for its prompt group
    (forecasting_agent.per_condition)."""
    base, variant = split_condition(condition)
    agents = fitted_models["variants"][variant] if variant else fitted_models
    name = "forecaster_pipeline" if prompt_group(base) == "full_pipeline" else "forecaster"
    return {"forecaster": agents[name], "critic": agents["critic"]}


def _load_models(agents: dict) -> None:
    """Load the forecaster's model. A critic on the same model needs nothing
    more; a critic on another model is loaded before each batch's critic
    phase (see the two-phase batches in _predict_batch)."""
    llm = getattr(agents.get("forecaster"), "llm", None)
    if llm is not None:
        llm.load()


def _predict_batch(
    condition: str,
    cfg: dict,
    batch_ids: list[int],
    test_tensors: dict[int, dict],
    fitted_models: dict,
    n_workers: int,
    postprocess: dict | None = None,
) -> dict:
    """Predictions for one batch of test patients: {"y_pred", "y_lower",
    "y_upper"} shaped (len(batch_ids), horizon_hours, n_variables), plus this
    batch's pre-critic plausibility "violations", values the critic
    "clipped", LLM "fallbacks" and per-variable "filled" counts."""
    base, variant = split_condition(condition)
    llm_agents = condition_agents(fitted_models, condition) if base in LLM_CONDITIONS else {}
    variables = [v["name"] for v in cfg["variables"]]
    fallbacks_before = llm_agents["forecaster"].n_fallbacks if base in LLM_CONDITIONS else 0
    filled_before = dict(llm_agents["forecaster"].n_filled) if base in LLM_CONDITIONS else {}
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]

    shape = (len(batch_ids), horizon_hours, len(variables))
    params = postprocess if postprocess is not None else postprocess_params(cfg)
    hard = np.array([hard_ranges(cfg)[var] for var in variables])
    y_raw = np.full(shape, np.nan)      # LLM forecasts before post-processing
    y_pred = np.full(shape, np.nan)
    y_lower = np.full(shape, np.nan)
    y_upper = np.full(shape, np.nan)
    total_precritic_violations = 0
    total_clipped = 0

    if base == "naive":
        fill = naive_fill(fitted_models.get("reference"), variables)
        for pi, sid in enumerate(batch_ids):
            r = naive_forecast(test_tensors[sid]["obs"], horizon_hours, fill)
            y_pred[pi] = r["forecast_array"]
            y_lower[pi] = r["forecast_array"] - r["halfwidth_array"][None, :]
            y_upper[pi] = r["forecast_array"] + r["halfwidth_array"][None, :]

    elif base in PERSISTENCE_BASELINES:
        fill = naive_fill(fitted_models.get("reference"), variables)
        window = int(cfg["baselines"].get("recent_mean_hours", 6))
        for pi, sid in enumerate(batch_ids):
            r = persistence_forecast(test_tensors[sid]["obs"], horizon_hours, base, window, fill)
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
            fcast = _clip_to_valid(r["forecast_array"], cfg)
            y_pred[pi] = fcast
            y_lower[pi] = fcast - r["halfwidth_array"][None, :]
            y_upper[pi] = fcast + r["halfwidth_array"][None, :]

    elif base == "lstm":
        lstm_model: LSTMBaseline = fitted_models["lstm"]
        if cfg.get("performance", {}).get("batch_predict_baselines", False) and hasattr(lstm_model, "predict_batch"):
            # One batched forward pass on the GPU instead of one Python-level
            # call per patient — this is where a 40GB card actually pays off
            # for the LSTM baseline at larger cohort sizes; a single-patient
            # forward pass barely uses the GPU at all.
            obs_stack = np.stack([test_tensors[sid]["obs"] for sid in batch_ids])
            batch = lstm_model.predict_batch(obs_stack)
            fcast = _clip_to_valid(batch["forecast_array"], cfg)
            y_pred[:] = fcast
            y_lower[:] = fcast - batch["halfwidth_array"][:, None, :]
            y_upper[:] = fcast + batch["halfwidth_array"][:, None, :]
        else:
            for pi, sid in enumerate(batch_ids):
                r = lstm_model.predict(test_tensors[sid]["obs"])
                fcast = _clip_to_valid(r["forecast_array"], cfg)
                y_pred[pi] = fcast
                y_lower[pi] = fcast - r["halfwidth_array"][None, :]
                y_upper[pi] = fcast + r["halfwidth_array"][None, :]

    elif base == "single_model_llm":
        # DT-GPT-style: same LLM as the pipeline, no similarity conditioning, no critic.
        forecaster: ForecastingAgent = llm_agents["forecaster"]

        def _single_model_worker(sid):
            obs = test_tensors[sid]["obs"]
            raw = forecaster.forecast(obs, similarity_context=None)
            fcast, interval = _llm_forecast_to_arrays(
                postprocess_forecast(raw, obs, variables, params, hard), variables, horizon_hours)
            return fcast, interval, _llm_forecast_to_arrays(raw, variables, horizon_hours)[0]

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Each Ollama HTTP call releases the GIL while waiting on I/O, so a
            # thread pool (not multiprocessing) is enough to get concurrent
            # requests in flight — see performance.llm_max_concurrent_requests
            # in the config and the matching OLLAMA_NUM_PARALLEL note in the
            # README. Results are collected via map(), which preserves order,
            # so y_pred[pi] still lines up with batch_ids[pi].
            for pi, (fcast, interval, raw_fcast) in enumerate(ex.map(_single_model_worker, batch_ids)):
                y_raw[pi] = raw_fcast
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
        critic_stage = critic_review_stage(cfg)

        def _forecast_worker(sid):
            sim_ctx = similarity.query(test_tensors[sid]["obs"]) if use_similarity else None
            return sid, forecaster.forecast(test_tensors[sid]["obs"], similarity_context=sim_ctx)

        def _critic_worker(sid_raw):
            sid, raw = sid_raw
            if use_critic and critic_stage == "raw":
                # The critic reviews the LLM's own output, then it is
                # post-processed. y_raw is the reviewed forecast, the one
                # post-processing starts from (calibrate.fit_postprocess).
                reviewed = critic.review(raw)
                raw = {**raw, "forecast": reviewed["forecast"]}
                final_forecast = postprocess_forecast(raw, test_tensors[sid]["obs"], variables, params, hard)
                fcast, interval = _llm_forecast_to_arrays(final_forecast, variables, horizon_hours)
                return (fcast, interval, reviewed["violations_before_correction"], reviewed["clipped_values"],
                        _llm_forecast_to_arrays(raw, variables, horizon_hours)[0])
            # Default (stage final): post-processing first, so the critic
            # reviews the final forecast.
            post = postprocess_forecast(raw, test_tensors[sid]["obs"], variables, params, hard)
            if use_critic:
                reviewed = critic.review(post)
                final_forecast = {"forecast": reviewed["forecast"], "interval_halfwidth": raw["interval_halfwidth"]}
                violations, clipped = reviewed["violations_before_correction"], reviewed["clipped_values"]
            else:
                final_forecast = post
                violations = clipped = 0

            fcast, interval = _llm_forecast_to_arrays(final_forecast, variables, horizon_hours)
            return fcast, interval, violations, clipped, _llm_forecast_to_arrays(raw, variables, horizon_hours)[0]

        # A critic on another model (critic_variant): forecast the whole batch,
        # then review it, so Ollama switches models twice per batch. Run per
        # patient, both models' requests interleave, and two models that don't
        # fit on the GPU together (MedGemma and Gemma 3, ~56 GiB each at 8
        # slots, job 1034109) are evicted back and forth, timing requests out.
        # Each patient's critic still sees only that patient's forecast.
        two_phase = (use_critic and critic.llm is not None
                     and critic.llm.model != forecaster.llm.model)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            if two_phase:
                # Load each phase's model first (up to llm_client.LOAD_TIMEOUT_S),
                # so the phase's first requests don't time out on the switch.
                forecaster.llm.load()
                raws = list(ex.map(_forecast_worker, batch_ids))
                critic.llm.load()
                results = ex.map(_critic_worker, raws)
            else:
                results = ex.map(lambda sid: _critic_worker(_forecast_worker(sid)), batch_ids)
            for pi, (fcast, interval, violations, clipped, raw_fcast) in enumerate(results):
                y_raw[pi] = raw_fcast
                y_pred[pi] = fcast
                y_lower[pi] = fcast - interval
                y_upper[pi] = fcast + interval
                total_precritic_violations += violations
                total_clipped += clipped

    else:
        raise ValueError(f"Unknown condition '{condition}'")

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "y_raw": y_raw if base in LLM_CONDITIONS else None,
        "violations": int(total_precritic_violations),
        "clipped": int(total_clipped),
        "fallbacks": int(llm_agents["forecaster"].n_fallbacks - fallbacks_before) if base in LLM_CONDITIONS else 0,
        "filled": np.array([llm_agents["forecaster"].n_filled[v] - filled_before[v] for v in variables]
                           if base in LLM_CONDITIONS else np.zeros(len(variables)), dtype=int),
    }


def ensembles(cfg: dict) -> dict[str, list[str]]:
    """`ensembles:` in the config, e.g. [[full_pipeline, lstm]], as
    {"full_pipeline+lstm": ["full_pipeline", "lstm"]}. An ensemble is not
    run: its forecast is the mean of its members' saved forecasts (and its
    interval half-width the mean of theirs), computed by calibrate.py and
    run_experiment.py after the members (Trillium tri_lean_exp2: Med42's
    pipeline averaged with the LSTM beat the LSTM alone, p = 0.0008)."""
    return {"+".join(members): list(members) for members in cfg.get("ensembles") or []}


def ensemble_result(members: list[dict]) -> dict:
    """The mean of member results (dicts with y_true, y_pred, y_lower,
    y_upper, stay_ids over the same patients in the same order)."""
    for m in members[1:]:
        if list(m["stay_ids"]) != list(members[0]["stay_ids"]):
            raise ValueError("ensemble members were run on different patients")
    y_pred = np.mean([m["y_pred"] for m in members], axis=0)
    halfwidth = np.mean([(m["y_upper"] - m["y_lower"]) / 2 for m in members], axis=0)
    return {"y_true": members[0]["y_true"], "y_pred": y_pred, "y_lower": y_pred - halfwidth,
            "y_upper": y_pred + halfwidth, "stay_ids": list(members[0]["stay_ids"])}


def validate_conditions(cfg: dict) -> None:
    """Fail at startup, not hours into a run, on a misspelled condition or an
    undefined / misplaced LLM variant."""
    known = ("naive", "gbm", "lstm") + PERSISTENCE_BASELINES + LLM_CONDITIONS
    critic_review_stage(cfg)                    # raises on an unknown critic_agent.stage
    validate_per_condition(cfg)
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
    for name, members in ensembles(cfg).items():
        missing = [m for m in members if m not in cfg["conditions"]]
        if len(members) < 2 or missing:
            raise ValueError(f"Ensemble '{name}' needs at least two members, all in `conditions` "
                             f"(missing: {missing})")


def fit_models(cfg: dict, train_tensors: dict[int, dict], checkpoint: RunCheckpoint | None = None) -> dict:
    """Fit every model that needs training/indexing once, shared across
    conditions (the LLM-based agents don't need "fitting" beyond loading).
    With `checkpoint`, the GBM and LSTM save their progress there and resume
    from it; the similarity index is cheap and simply refitted."""
    variables = [v["name"] for v in cfg["variables"]]
    horizon_hours = cfg["cohort"]["forecast_horizon_hours"]
    # The training cohort's per-variable median/IQR: fills variables a
    # patient has no observation of (naive, LLM fallback) and anchors prompts.
    fitted = {"reference": reference_stats(train_tensors, variables)}

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
        fitted["lstm"] = LSTMBaseline(variables, horizon_hours, hidden=lstm_hidden,
                                      input_mode=cfg["baselines"].get("lstm_input", "zero_fill")).fit(
            train_tensors, epochs=lstm_epochs, lr=cfg["baselines"].get("lstm_learning_rate", 1e-3),
            checkpoint_path=lstm_path,
            checkpoint_every=cfg["baselines"].get("lstm_checkpoint_every_epochs", 20),
            seed=cfg["baselines"].get("lstm_seed"),
        )

    in_use = llm_variants_in_use(cfg)
    needs_default_llm = None in in_use
    used_variants = [v for v in in_use if v is not None]

    if needs_default_llm:
        fitted.update(_build_llm_agents(cfg, None, fitted["reference"]))
    fitted["variants"] = {}
    for variant in used_variants:
        fitted["variants"][variant] = _build_llm_agents(cfg, variant, fitted["reference"])
        llm = fitted["variants"][variant]["forecaster"].llm
        log.info("LLM variant '%s' -> model %s (pulled as %s)", variant, llm.model, llm.source_model)
    if needs_default_llm or used_variants:
        fitted["similarity"] = SimilarityAgent(variables, k=cfg["similarity_agent"]["k_neighbors"]).fit(train_tensors)

    return fitted


def _build_llm_agents(cfg: dict, variant: str | None, reference: dict | None = None) -> dict:
    """Forecaster and critic for one LLM variant (None = llm.model). They
    share one LocalLLM, unless the variant sets `critic_variant`: then the
    critic gets its own, built from that variant's llm settings."""
    vcfg = cfg_for_llm_variant(cfg, variant)
    llm = LocalLLM(vcfg)
    cv = critic_variant(cfg, variant)
    critic_llm = LocalLLM(cfg_for_llm_variant(cfg, cv)) if cv is not None else llm
    if cv is not None:
        log.info("LLM variant '%s': critic uses variant '%s' (model %s).", variant, cv, critic_llm.model)
    # One forecaster per prompt group; they are the same unless
    # forecasting_agent.per_condition sets something, and share the LLM.
    return {
        "forecaster": ForecastingAgent(llm, cfg_for_prompt_group(vcfg, "single_model_llm"), reference),
        "forecaster_pipeline": ForecastingAgent(llm, cfg_for_prompt_group(vcfg, "full_pipeline"), reference),
        "critic": CriticAgent(critic_llm, vcfg, enabled=vcfg["critic_agent"]["enabled"]),
    }
