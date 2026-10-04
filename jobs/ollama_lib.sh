#!/bin/bash
# =============================================================================
# Ollama helpers shared by GPU jobs — SOURCED, not submitted.
#
#   source "$DT_REPO/jobs/ollama_lib.sh"
#   start_ollama "$CONFIG" [tag]      # per-job port, stopped when the job exits;
#                                     # log: $DT_LOG_DIR/ollama-<job id>[-<tag>].log
#   stop_ollama                       # stop it early (e.g. to restart with other settings)
#   require_models "$CONFIG" all      # every model the config's conditions use
#   require_models "$CONFIG" default  # only llm.model (e.g. for tuning)
#
# start_ollama runs `ollama serve` on 127.0.0.1 at a port derived from the job
# id — not the default 11434, which another user's server may hold on a shared
# node — and exports OLLAMA_HOST (for the ollama CLI) and DT_OLLAMA_HOST (read
# by src/llm_client.py instead of llm.ollama_host). It waits for the server
# with curl retries (no sleep) and installs an EXIT trap that stops it.
# The server's settings come from the config's performance section:
#   OLLAMA_NUM_PARALLEL     = performance.llm_max_concurrent_requests
#   OLLAMA_FLASH_ATTENTION  = performance.ollama_flash_attention (default off)
# so the request slots always match the requests the pipeline sends at once.
# A benchmark can override both: start_ollama "$CONFIG" <tag> <parallel> <0|1>.
# Needs the venv's python (PyYAML) and OLLAMA_MODELS (paths.ollama_models in your profile).
# =============================================================================

start_ollama() {
    local config=${1:?start_ollama needs the config file} tag=${2:-} settings np fa
    local port=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))
    local log_file="$DT_LOG_DIR/ollama-${SLURM_JOB_ID:-local}${tag:+-$tag}.log"
    settings=$(python -c 'import sys; sys.path.insert(0, "src")
from common import load_config, ollama_server_settings
s = ollama_server_settings(load_config(sys.argv[1]))
print(s["OLLAMA_NUM_PARALLEL"], s["OLLAMA_FLASH_ATTENTION"])' "$config") || return 1
    read -r np fa <<< "$settings"
    export OLLAMA_NUM_PARALLEL="${3:-$np}" OLLAMA_FLASH_ATTENTION="${4:-$fa}"
    echo "== ollama settings: OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL OLLAMA_FLASH_ATTENTION=$OLLAMA_FLASH_ATTENTION" \
         "(${3:+overridden; }config: $np / $fa from $config) =="
    export OLLAMA_HOST="127.0.0.1:${port}"
    export DT_OLLAMA_HOST="http://127.0.0.1:${port}"
    mkdir -p "$OLLAMA_MODELS" "$DT_LOG_DIR"
    ollama serve > "$log_file" 2>&1 &
    OLLAMA_PID=$!
    trap stop_ollama EXIT
    echo "== waiting for ollama on ${OLLAMA_HOST} =="
    if ! curl -sf --retry 60 --retry-delay 1 --retry-connrefused "http://${OLLAMA_HOST}/api/tags" > /dev/null; then
        echo "== ollama did not become ready — see $log_file ==" >&2
        return 1
    fi
    echo "== ollama ready =="
}

stop_ollama() {
    [ -n "${OLLAMA_PID:-}" ] || return 0
    echo "== stopping ollama (pid $OLLAMA_PID) at $(date) =="
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
}

# require_models <config> <all|default>: abort if a needed model isn't in
# $OLLAMA_MODELS (compute nodes shouldn't pull mid-job; run prep2 first).
require_models() {
    local config=$1 which=$2 models pulled model
    models=$(python -c 'import sys; sys.path.insert(0, "src")
from common import load_config, llm_models_in_use, ollama_model_name
cfg = load_config(sys.argv[1])
print("\n".join(llm_models_in_use(cfg) if sys.argv[2] == "all" else [ollama_model_name(cfg["llm"])]))' "$config" "$which") || return 1
    pulled=$(ollama list | awk 'NR > 1 {print $1}')
    for model in $models; do
        if ! grep -Fxq "$model" <<< "$pulled"; then
            echo "== model '$model' not found in \$OLLAMA_MODELS ($OLLAMA_MODELS) — run first:" >&2
            echo "==   sbatch jobs/prep2_download_models.sh $config" >&2
            return 1
        fi
        echo "== model '$model' found =="
    done
}
