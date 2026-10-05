"""
Thin wrapper around a locally-hosted, open-weight LLM. Talks only to
127.0.0.1 (the local Ollama server) — this is the one and only "network" call
anywhere in the pipeline, and it never leaves the machine.

Install Ollama (https://ollama.com) and pull a model once, e.g.:
    ollama pull llama3.1:8b-instruct-q4_K_M
then start the server (usually auto-started as a background service):
    ollama serve

If you'd rather not run Ollama, swap this module's implementation for
llama-cpp-python against a local .gguf file — the rest of the pipeline only
depends on LocalLLM.generate(prompt) -> str, so nothing else needs to change.
"""
from __future__ import annotations

import json
import os
import time

import requests
from requests.adapters import HTTPAdapter

from common import get_logger, ollama_model_name

log = get_logger("llm_client")

# Extra attempts after an HTTP 5xx from Ollama before the call fails.
SERVER_ERROR_RETRIES = 2
# Seconds to wait for Ollama to load a model (LocalLLM.load). Loading
# Baichuan-M2-32B from $SCRATCH on Trillium took ~176 s (job 1033530), so the
# first requests of a condition, at llm.request_timeout_s=180, timed out while
# the model loaded (calibrate job 1032638) and were scored as fallbacks. A
# constant, not an llm.* key, because the llm section is part of every LLM
# condition's checkpoint fingerprint.
LOAD_TIMEOUT_S = 900


class LocalLLM:
    def __init__(self, cfg: dict):
        llm_cfg = cfg["llm"]
        self.backend = llm_cfg["backend"]
        # DT_OLLAMA_HOST lets a job point at its own per-job Ollama port
        # (another user's server may hold the default 11434 on a shared node).
        self.host = os.environ.get("DT_OLLAMA_HOST") or llm_cfg["ollama_host"]
        self.model = ollama_model_name(llm_cfg)      # alias if set — the name sent to Ollama
        self.source_model = llm_cfg["model"]         # the tag it was pulled as
        self.temperature = llm_cfg["temperature"]
        self.max_tokens = llm_cfg["max_tokens"]
        self.timeout = llm_cfg["request_timeout_s"]
        self.num_ctx = llm_cfg["num_ctx"]
        # Ollama's `think` for reasoning models: false turns the reasoning
        # off. Absent = not sent (Ollama's default). Baichuan-M2 reasoned past
        # max_tokens on every forecast, leaving the answer empty (job 1033722).
        self.think = llm_cfg.get("think")

        # How many forecasting/critic calls src/pipeline.py will fire at this
        # LocalLLM concurrently (see performance.llm_max_concurrent_requests
        # in the config). This doesn't change what a single call does — it
        # just sizes the HTTP connection pool so concurrent calls from a
        # thread pool don't serialize on socket setup. Actual concurrent
        # *generation* on the Ollama side also needs the server started with
        # OLLAMA_NUM_PARALLEL >= this value (see README) and enough VRAM to
        # hold that many copies of the model's KV cache at num_ctx — on a
        # single 40GB card with a 30B+ model, 2-4 is usually the practical
        # ceiling, not the CPU/RAM-driven concurrency used elsewhere in the
        # pipeline (e.g. cohort extraction, GBM/LSTM training).
        pool_size = max(1, int(cfg.get("performance", {}).get("llm_max_concurrent_requests", 1)))
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        if self.backend != "ollama":
            raise NotImplementedError(
                f"backend '{self.backend}' not implemented in this reference version; "
                "implement LocalLLM.generate() against llama-cpp-python if you need it."
            )
        self._check_server()

    def _check_server(self) -> None:
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                setup = f"ollama pull {self.source_model}"
                if self.model != self.source_model:
                    setup += f" && ollama cp {self.source_model} {self.model}"
                log.warning(
                    "Configured model '%s' not found in `ollama list` output (%s). "
                    "Run `%s` before starting the experiment.",
                    self.model, models, setup,
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not reach the local Ollama server at {self.host}. "
                "Install Ollama and run `ollama serve` (or check it's already running), "
                "then `ollama pull <model>` for the model named in the config."
            )

    def load(self) -> None:
        """Have Ollama load the model now (a request with no prompt loads it
        without generating), so the first forecasts don't spend their
        request timeout waiting for it. Raises if it can't be loaded.
        num_ctx must match generate()'s: without it Ollama picks its own
        context length, and at 8 request slots Baichuan-M2's filled the
        job's 188 GiB of host memory (OOM-killed, job 1033643); a different
        num_ctx would also make the first real request reload the model."""
        t0 = time.time()
        resp = self._session.post(f"{self.host}/api/generate", timeout=LOAD_TIMEOUT_S,
                                  json={"model": self.model, "options": {"num_ctx": self.num_ctx}})
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} loading {self.model} in Ollama: {resp.text[:300]}", response=resp)
        log.info("Model %s loaded in %.0fs.", self.model, time.time() - t0)

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        """json_mode asks for any valid JSON; schema (a JSON schema) also
        constrains decoding to that shape, e.g. a list's exact length."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = bool(self.think)
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"

        t0 = time.time()
        # Ollama answers some requests with a 500 that a repeat of the same
        # request doesn't hit (job 23150606: 61 of Med42's 80 fallbacks).
        # Retry those, so a server hiccup isn't scored as a model failure.
        for attempt in range(1 + SERVER_ERROR_RETRIES):
            resp = self._session.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            if resp.status_code < 500:
                break
            log.warning("Ollama returned HTTP %d (attempt %d of %d): %s", resp.status_code,
                        attempt + 1, 1 + SERVER_ERROR_RETRIES, resp.text[:300])
        if not resp.ok:
            # raise_for_status() alone drops the body, which is the only place
            # Ollama says what went wrong (its server log doesn't).
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} from Ollama /api/generate: {resp.text[:300]}", response=resp)
        body = resp.json()
        text = body["response"]
        log.debug("LLM call took %.2fs, %d chars out.", time.time() - t0, len(text))
        # Ollama reports done_reason "length" when generation hit num_predict:
        # the output is cut off mid-answer, so say so rather than letting the
        # caller report it as malformed JSON.
        # The start and end of the output show where the tokens went (e.g. a
        # runaway list or digits), which the token count alone doesn't.
        if body.get("done_reason") == "length":
            raise ValueError(f"LLM output truncated at max_tokens={self.max_tokens} "
                             f"({body.get('eval_count', '?')} tokens generated) — raise llm.max_tokens in the config; "
                             f"output starts: {text[:150]!r}, ends: {text[-150:]!r}; "
                             f"{len(body.get('thinking') or '')} chars of reasoning (Ollama's thinking field)")
        return text


def extract_json_block(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences even when asked not to;
    pull out the first {...} block and parse it, raising a clear error if that
    fails so the caller can retry or fall back."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
