#!/bin/bash
# =============================================================================
# Ollama helpers shared by GPU jobs — SOURCED, not submitted.
#
#   source "$DT_REPO/jobs/ollama_lib.sh"
#   start_ollama "$CONFIG" [tag]      # per-job port, stopped when the job exits;
#                                     # log: $DT_LOG_DIR/ollama-<job id>[-<tag>].log
#   stop_ollama                       # stop it early (e.g. to restart with other settings)
#   require_models "$CONFIG" all      # every model the config's conditions use
#   require_models "$CONFIG" default  # only llm.model
#   require_models "$CONFIG" tune [--grid <file>]   # the models the tuning grid's rounds use
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
    # ollama serve keeps its key in ~/.ollama. Where compute nodes can't write
    # $HOME (Trillium) and no key was made on a login node yet, give it the
    # job's own directory instead (models still come from OLLAMA_MODELS).
    local serve_home=$HOME
    if [ ! -f "$HOME/.ollama/id_ed25519" ] && [ ! -w "$HOME" ]; then
        serve_home=${SLURM_TMPDIR:-/tmp}
    fi
    # Ollama gives each CUDA library 30 s to load during GPU discovery. Read
    # from a networked filesystem that can time out, and Ollama then quietly
    # runs on Vulkan (Fir job 63850581: 1.65x slower) or the CPU. So: read
    # the libraries once first (they then load from the page cache), turn
    # Vulkan off, and check below that the GPU was found through CUDA.
    local libdir
    libdir="$(dirname "$(dirname "$(command -v ollama)")")/lib/ollama"
    if [ -d "$libdir" ]; then
        echo "== reading Ollama's libraries ($libdir) into the page cache =="
        find "$libdir" -maxdepth 2 -name '*.so*' -type f -exec cat {} + > /dev/null 2>&1 || true
    fi
    export OLLAMA_VULKAN=false
    HOME=$serve_home ollama serve > "$log_file" 2>&1 &
    OLLAMA_PID=$!
    trap stop_ollama EXIT
    echo "== waiting for ollama on ${OLLAMA_HOST} =="
    if ! curl -sf --retry 180 --retry-delay 1 --retry-connrefused "http://${OLLAMA_HOST}/api/tags" > /dev/null; then
        echo "== ollama did not become ready — see $log_file ==" >&2
        return 1
    fi
    check_ollama_gpu "$log_file" || return 1
    echo "== ollama ready =="
}

# The backend Ollama found the GPU with ("inference compute" in its log);
# fails unless it is CUDA, so the job stops (and run_all resubmits it)
# instead of running hours on Vulkan or the CPU. DT_OLLAMA_REQUIRE_CUDA=0
# skips the check, e.g. on a machine without an NVIDIA GPU.
check_ollama_gpu() {
    local log_file=$1 line fd tail_pid
    [ "${DT_OLLAMA_REQUIRE_CUDA:-1}" = 0 ] && return 0
    # Waits for the line as it is written (no polling), at most 120 s; the
    # tail is stopped as soon as grep has it.
    exec {fd}< <(timeout ${DT_OLLAMA_GPU_WAIT_S:-120} tail -n +1 -F "$log_file" 2>/dev/null)
    tail_pid=$!
    line=$(grep -m1 'msg="inference compute"' <&"$fd") || true      # no match: reported below
    exec {fd}<&-
    kill "$tail_pid" 2>/dev/null || true
    if [[ "$line" == *"library=CUDA"* ]]; then
        echo "== ollama found the GPU through CUDA =="
        return 0
    fi
    echo "== ollama did not find the GPU through CUDA (${line:-no 'inference compute' line in ${DT_OLLAMA_GPU_WAIT_S:-120} s}) — stopping;" \
         "see $log_file (GPU discovery timeouts?) ==" >&2
    return 1
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
import yaml
from common import load_config, llm_models_for_tuning, llm_models_in_use, ollama_model_name
cfg = load_config(sys.argv[1])
if sys.argv[2] == "all":
    models = llm_models_in_use(cfg)
elif sys.argv[2] == "tune":
    # The grid tune.py will use: --grid among the job arguments, else the
    # config'"'"'s tuning_grid, else the standard one.
    grid, args = cfg.get("tuning_grid") or "config/tuning_grid.yaml", sys.argv[3:]
    for i, a in enumerate(args):
        if a == "--grid" and i + 1 < len(args):
            grid = args[i + 1]
        elif a.startswith("--grid="):
            grid = a.split("=", 1)[1]
    models = llm_models_for_tuning(cfg, yaml.safe_load(open(grid))["tuning"])
else:
    models = [ollama_model_name(cfg["llm"])]
print("\n".join(models))' "$config" "$which" "${@:3}") || return 1
    [ -n "$models" ] || { echo "== no LLM needed ($which) =="; return 0; }
    pulled=$(ollama list | awk 'NR > 1 {print $1}')
    for model in $models; do
        if ! grep -Fxq "$model" <<< "$pulled"; then
            echo "== model '$model' not found in \$OLLAMA_MODELS ($OLLAMA_MODELS) — run first:" >&2
            echo "==   bash jobs/submit.sh jobs/prep2_download_models.sh $config" >&2
            return 1
        fi
        echo "== model '$model' found =="
    done
}
