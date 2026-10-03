#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 3/4: src/run_experiment.py
# The GPU step: fits the naive/GBM/LSTM baselines and runs every configured
# condition, including the LLM-based conditions (single_model_llm,
# full_pipeline*), against the local Ollama server. Requires steps 1 and 2
# to already be done (item_mapping.json + cohort/panel parquet files).
#
# Settings (project directory, data_root, Ollama model store) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments. Submit from the project directory.
#
# Works for either scope — pass the config file as the first argument:
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
# --time=08:00:00 below is sized for the slower 19-variable scope; the
# 5-variable job will typically finish well inside that.
#
# Resuming: run_experiment.py checkpoints as it goes (GBM per variable, LSTM
# every few epochs, each condition per batch of
# performance.checkpoint_batch_size test patients). If this job hits its time
# limit or is stopped, submit it again with the same config and it continues
# where it left off, losing at most one batch. To queue the continuation up
# front, chain it — it starts when the first job ends, however it ends, and
# exits quickly if there's nothing left to do:
#   JOB=$(sbatch --parsable jobs/step3_run_experiment_nibi.sh config/config_nibi_lean.yaml)
#   sbatch --dependency=afterany:$JOB jobs/step3_run_experiment_nibi.sh config/config_nibi_lean.yaml
# Arguments after the config go to run_experiment.py; to discard the
# checkpoints and start from scratch (e.g. after changing prompts or code):
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_nibi_lean.yaml --full-refresh
#
# Before the first submission of EITHER scope (and after adding a model or
# alias to the config), download the models it needs with the same config:
#   sbatch jobs/prep2_download_models.sh config/config_nibi_lean.yaml
# This job doesn't pull models itself: it aborts early with a clear message,
# listing the setup commands, if any model or alias isn't found.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step3-experiment
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Settings come from the profile (~/.config/dt_profile.yml, or --profile
# <file> among this job's arguments) — see jobs/load_profile.sh. The
# remaining arguments are this job's own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the project directory: cd <project> && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

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

# Every model the config uses (llm.model plus any llm.variants named in
# `conditions`) must already be pulled (see header above) — compute nodes
# typically can't reach the internet to pull them now.
MODELS=$(python -c 'import sys; sys.path.insert(0, "src"); from common import load_config, llm_models_in_use; print("\n".join(llm_models_in_use(load_config(sys.argv[1]))))' "$CONFIG")
PULLED=$(ollama list | awk 'NR > 1 {print $1}')
for MODEL in $MODELS; do
    if ! grep -Fxq "$MODEL" <<< "$PULLED"; then
        echo "== model '$MODEL' not found in \$OLLAMA_MODELS ($OLLAMA_MODELS) —" >&2
        echo "== run these on a login node before submitting this job. Aborting. ==" >&2
        python -c 'import sys; sys.path.insert(0, "src"); from common import load_config, llm_setup_commands; print("\n".join(llm_setup_commands(load_config(sys.argv[1]))))' "$CONFIG" >&2
        exit 1
    fi
    echo "== model '$MODEL' found =="
done

echo "== step 3/4: run_experiment.py =="
python src/run_experiment.py --config-file "$CONFIG" "${@:2}"   # e.g. --full-refresh

echo "== job $SLURM_JOB_ID finished at $(date) =="
# `trap cleanup EXIT` stops ollama on the way out, success or failure.
