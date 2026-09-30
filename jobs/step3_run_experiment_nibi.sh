#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 3/4: src/run_experiment.py
# The GPU step: fits the naive/GBM/LSTM baselines and runs every configured
# condition, including the LLM-based conditions (single_model_llm,
# full_pipeline*), against the local Ollama server. Requires steps 1 and 2
# to already be done (item_mapping.json + cohort/panel parquet files).
#
# Reads the project directory from $AGENTIC_DT_PRJ and the Ollama model
# store from $OLLAMA_MODELS (both must be exported in the submitting shell
# before `sbatch` — sbatch passes the submission environment through by
# default; $OLLAMA_MODELS is also the standard Ollama env var name, so the
# same export works for `ollama pull`/`ollama serve` run by hand).
#
# Works for either scope — pass the config file as the first argument:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   export OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama_local/models
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
# --time=08:00:00 below is sized for the slower 19-variable scope; the
# 5-variable job will typically finish well inside that.
#
# Before the first submission of EITHER scope, on a login node (compute
# nodes typically have no internet access):
#   export OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama_local/models
#   ollama pull qwen2.5:32b-instruct-q8_0
# This job aborts early with a clear message if the model isn't found,
# rather than trying (and failing) to pull it mid-job.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step3-experiment
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${AGENTIC_DT_PRJ:?AGENTIC_DT_PRJ is not set — export it to the project directory before sbatch, e.g. export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1}"
: "${OLLAMA_MODELS:?OLLAMA_MODELS is not set — export it to your Ollama model store before sbatch, e.g. export OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama_local/models}"
PROJECT_DIR="$AGENTIC_DT_PRJ"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config_nibi_lean.yaml}"
OLLAMA_PORT=11434
export OLLAMA_NUM_PARALLEL=4        # must match the config's performance.llm_max_concurrent_requests
export OLLAMA_MODELS         # already set by the submitting shell (checked above) — re-exported for ollama serve/list below

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  project_dir=$PROJECT_DIR  config=$CONFIG =="

if [[ "$CONFIG" == *full_variables* ]]; then
    echo "=============================================================="
    echo " ALTERNATE SCOPE: 19-variable panel, NOT covered by current"
    echo " UVic HREB approval. See config/config_nibi_full_variables.yaml"
    echo " and README.md 'Two configs, two scopes' before trusting these"
    echo " results for anything beyond code readiness."
    echo "=============================================================="
fi

module load python/3.11
source "$PROJECT_DIR/.venv/bin/activate"

# --- start Ollama in the background, on this node only (127.0.0.1) --------
mkdir -p "$OLLAMA_MODELS"
ollama serve > "ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    echo "== stopping ollama (pid $OLLAMA_PID) at $(date) =="
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for the server to be ready before touching it --------------------
echo "== waiting for ollama on 127.0.0.1:${OLLAMA_PORT} =="
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" > /dev/null; then
        echo "== ollama ready after ${i}s =="
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "== ollama did not become ready within 60s — aborting ==" >&2
        exit 1
    fi
    sleep 1
done

# Model must already be pulled (see header above) — compute nodes typically
# can't reach the internet to pull it now.
MODEL="qwen2.5:32b-instruct-q8_0"
if ! ollama list | grep -q "$MODEL"; then
    echo "== model '$MODEL' not found in \$OLLAMA_MODELS ($OLLAMA_MODELS) —" >&2
    echo "== pull it on a login node before submitting this job. Aborting. ==" >&2
    exit 1
fi

echo "== step 3/4: run_experiment.py =="
python src/run_experiment.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
# `trap cleanup EXIT` stops ollama on the way out, success or failure.
