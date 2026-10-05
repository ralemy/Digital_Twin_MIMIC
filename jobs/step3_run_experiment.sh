#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 3/4: src/run_experiment.py
# The GPU step: fits the naive/GBM/LSTM baselines and runs every configured
# condition, including the LLM-based conditions (single_model_llm,
# full_pipeline*), against the local Ollama server. Requires steps 1 and 2
# to already be done (item_mapping.json + cohort/panel parquet files).
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base:
#
# Works for either scope — pass the config file as the first argument:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   bash jobs/submit.sh jobs/step3_run_experiment.sh config/config_alliance_lean.yaml
#   bash jobs/submit.sh jobs/step3_run_experiment.sh config/config_alliance_full.yaml
# Defaults to config_alliance_lean.yaml if omitted.
# --time=08:00:00 below does NOT fit a whole run of either scope (see
# docs/runtime_estimates.md): with every condition in the config, the
# 5-variable run takes ~13.5 GPU-hours (2 chained jobs) and the 19-variable
# run ~49 (7 chained jobs). Chain the continuations as shown below; shorter
# jobs schedule sooner and nothing is lost but the batch in progress.
#
# Resuming: run_experiment.py checkpoints as it goes (GBM per variable, LSTM
# every few epochs, each condition per batch of
# performance.checkpoint_batch_size test patients). If this job hits its time
# limit or is stopped, submit it again with the same config and it continues
# where it left off, losing at most one batch. To queue the continuation up
# front, chain it — it starts when the first job ends, however it ends, and
# exits quickly if there's nothing left to do:
#   JOB=$(bash jobs/submit.sh --parsable jobs/step3_run_experiment.sh config/config_alliance_lean.yaml)
#   bash jobs/submit.sh --dependency=afterany:$JOB jobs/step3_run_experiment.sh config/config_alliance_lean.yaml
# Arguments after the config go to run_experiment.py; to discard the
# checkpoints and start from scratch (e.g. after changing prompts or code):
#   bash jobs/submit.sh jobs/step3_run_experiment.sh config/config_alliance_lean.yaml --full-refresh
#
# Before the first submission of EITHER scope (and after adding a model or
# alias to the config), download the models it needs with the same config:
#   bash jobs/submit.sh jobs/prep2_download_models.sh config/config_alliance_lean.yaml
# This job doesn't pull models itself: it aborts early with a clear message,
# listing the setup commands, if any model or alias isn't found.
# =============================================================================
#SBATCH --job-name=mimic-twin-step3-experiment
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

# Settings come from your profile, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"

CONFIG="${1:-config/config_alliance_lean.yaml}"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  repo=$DT_REPO  config=$CONFIG =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"

# --- Ollama on a per-job port, on this node only (127.0.0.1) ---------------
# OLLAMA_NUM_PARALLEL / OLLAMA_FLASH_ATTENTION come from the config's
# performance section (see jobs/ollama_lib.sh).
# Every model the config uses (llm.model plus any llm.variants named in
# `conditions`) must already be pulled (see header above) — compute nodes
# typically can't reach the internet to pull them now.
source jobs/ollama_lib.sh
start_ollama "$CONFIG"
require_models "$CONFIG" all

echo "== step 3/4: run_experiment.py =="
python src/run_experiment.py --config-file "$CONFIG" "${@:2}"   # e.g. --full-refresh

echo "== job $SLURM_JOB_ID finished at $(date) =="
# start_ollama's EXIT trap stops ollama on the way out, success or failure.
