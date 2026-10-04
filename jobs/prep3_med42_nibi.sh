#!/bin/bash
# =============================================================================
# Slurm batch job — one-off check that Med42-70B loads and answers on a GPU
# node via Ollama, before wiring it into the experiment config.
#
# Paths (repo, MIMIC-IV, Ollama models, modules) come from
# jobs/setup_bash.sh, picked by cluster.
#
# Download the model and create its med42:70b alias first: enable the
# llama_Med42_70b conditions in the config, then
#   sbatch jobs/prep2_download_models.sh config/config_nibi_lean.yaml
#
# Then, from the repository base:
#   cd "$DT_REPO"     # with jobs/setup_bash.sh sourced (sets DT_REPO, SBATCH_ACCOUNT)
#   sbatch jobs/prep3_med42_nibi.sh                # tests med42:70b
#   sbatch jobs/prep3_med42_nibi.sh <model-name>   # tests another pulled model
# =============================================================================
#SBATCH --job-name=test-med42
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Paths come from jobs/setup_bash.sh, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

command -v ollama > /dev/null || { echo "== ollama not found in $DT_OLLAMA_BIN (DT_OLLAMA_BIN, jobs/setup_bash.sh) — install it per installing_ollama.md ==" >&2; exit 1; }

MODEL="${1:-med42:70b}"
OLLAMA_PORT=11434
export OLLAMA_MODELS

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== model=$MODEL  OLLAMA_MODELS=$OLLAMA_MODELS =="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# --- start Ollama in the background, on this node only (127.0.0.1) --------
ollama serve > "ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    echo "== stopping ollama (pid $OLLAMA_PID) at $(date) =="
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for the server to be ready (curl retries; no sleep in the job) ---
echo "== waiting for ollama on 127.0.0.1:${OLLAMA_PORT} =="
if ! curl -sf --retry 60 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null; then
    echo "== ollama did not become ready — see ollama-${SLURM_JOB_ID}.log. Aborting. ==" >&2
    exit 1
fi
echo "== ollama ready =="

# --- the model must already be pulled (compute nodes can't pull) ----------
if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
    echo "== model '$MODEL' not found in \$OLLAMA_MODELS ($OLLAMA_MODELS) —" >&2
    echo "== pull it on a login node before submitting this job. Aborting. ==" >&2
    ollama list >&2
    exit 1
fi
echo "== model '$MODEL' found =="

# --- run a prompt; --verbose prints load time and tokens/s -----------------
echo "== prompt 1 (includes model load time) =="
ollama run --verbose "$MODEL" "What are the first-line treatments for sepsis? Answer in 5 bullet points."

# Confirm the model is fully on the GPU: PROCESSOR should read "100% GPU".
# Anything showing CPU means it partly spilled out of VRAM and will be slow.
echo "== ollama ps =="
ollama ps
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "== prompt 2 (model already loaded — steady-state speed) =="
ollama run --verbose "$MODEL" "A 68-year-old ICU patient has MAP 58 mmHg, lactate 4.1 mmol/L and heart rate 118. Briefly, what is the likely problem and immediate next step?"

echo "== job $SLURM_JOB_ID finished at $(date) =="
# `trap cleanup EXIT` stops ollama on the way out, success or failure.
