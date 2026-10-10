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
from pathlib import Path

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
LOAD_TIMEOUT_S = 900   # jobs/ollama_lib.sh gives Ollama's own limit (OLLAMA_LOAD_TIMEOUT) a bit less


def model_blobs(model: str, models_dir: str | None = None) -> list[Path]:
    """The weight files ("model" layers) of an Ollama model in
    $OLLAMA_MODELS, from its manifest: e.g. llama3:70b-instruct-q4_K_M ->
    manifests/registry.ollama.ai/library/llama3/70b-instruct-q4_K_M,
    hf.co/<user>/<repo>:<tag> -> manifests/hf.co/<user>/<repo>/<tag>."""
    root = Path(models_dir or os.environ.get("OLLAMA_MODELS", ""))
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library"] + parts
    elif len(parts) == 2:
        parts = ["registry.ollama.ai"] + parts
    layers = json.loads((root / "manifests" / Path(*parts) / (tag or "latest")).read_text())["layers"]
    return [root / "blobs" / layer["digest"].replace(":", "-")
            for layer in layers if "model" in layer.get("mediaType", "")]


def preread(paths: list[Path], chunk: int = 64 << 20) -> int:
    """Read files start to end (into the page cache); returns bytes read."""
    total = 0
    for path in paths:
        with open(path, "rb", buffering=0) as f:
            while block := f.read(chunk):
                total += len(block)
    return total


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
        self._preread()
        t0 = time.time()
        resp = self._session.post(f"{self.host}/api/generate", timeout=LOAD_TIMEOUT_S,
                                  json={"model": self.model, "options": {"num_ctx": self.num_ctx}})
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} loading {self.model} in Ollama: {resp.text[:300]}", response=resp)
        log.info("Model %s loaded in %.0fs.", self.model, time.time() - t0)

    def _preread(self) -> None:
        """Read the model's weight file sequentially before Ollama loads it,
        unless it's loaded already. Ollama memory-maps the file, and on a
        networked filesystem the scattered reads of a model the node hasn't
        read yet can outlast its load limit: Rorqual, Llama-3-70B from
        /scratch, 5 min and not loaded (job 22936019). One sequential read is
        far faster, and the load then comes from the page cache.
        DT_OLLAMA_PREREAD=0 turns it off; any failure here only logs."""
        if os.environ.get("DT_OLLAMA_PREREAD", "1") == "0":
            return
        try:
            running = self._session.get(f"{self.host}/api/ps", timeout=30).json().get("models") or []
            if any(self.model in (m.get("name"), m.get("model")) for m in running):
                return
            t0 = time.time()
            n = preread(model_blobs(self.model))
            log.info("Read %s's weights (%.1f GB) in %.0fs before loading it.", self.model, n / 1e9, time.time() - t0)
        except Exception as e:  # noqa: BLE001 - an optimisation; the load itself still runs
            log.warning("Could not pre-read %s's weights (%s); loading it directly.", self.model, e)

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
        # A request whose prompt + answer filled the context window doesn't
        # fail in Ollama: llama-server runs with --context-shift and silently
        # drops early tokens, so the answer may have been written without
        # part of the prompt. Fail it instead (the caller falls back and
        # counts it). Lean requests used at most 2,187 of 8,192 tokens on
        # Trillium (tri_lean_exp2); a 19-variable Med42 request is expected
        # at ~7,500 of its 8,192. prompt_eval_count can undercount a prompt
        # whose start was reused from the cache, so a request close to the
        # limit may slip through: the full-scope probe checks the margin.
        used = int(body.get("prompt_eval_count") or 0) + int(body.get("eval_count") or 0)
        if used >= self.num_ctx:
            raise ValueError(f"LLM context overflow: prompt + answer used {used} tokens of num_ctx="
                             f"{self.num_ctx} ({body.get('prompt_eval_count', '?')} + "
                             f"{body.get('eval_count', '?')}); Ollama shifts the context silently, "
                             "so this answer may not have seen the whole prompt — raise llm.num_ctx "
                             "if the model allows it, or shorten the prompt")
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
