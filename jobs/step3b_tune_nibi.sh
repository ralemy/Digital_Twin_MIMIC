#!/bin/bash
# =============================================================================
# Slurm batch job — hyperparameter search (src/tune.py) on the TUNING subset
# of the validation split. Requires steps 1 and 2 (cohort + panel). Never
# reads the test split.
#
# Runs the rounds in the grid file (default config/tuning_grid.yaml; see its
# header for what is searched and the cost, ~6 GPU-hours from scratch) and
# writes config/<config stem>_tuned.yaml with the winning settings, plus
# <results_dir>/tuning/{trials.csv,rounds.json,tuned_overrides.yaml}.
#
# Resuming: every setting is checkpointed per batch of patients and cached
# once scored, so after a time limit or failure just submit it again — or
# chain a continuation up front:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   JOB=$(sbatch --parsable jobs/step3b_tune_nibi.sh config/config_alliance_lean.yaml)
#   sbatch --dependency=afterany:$JOB jobs/step3b_tune_nibi.sh config/config_alliance_lean.yaml
# Other arguments after the config go to tune.py:
#   sbatch jobs/step3b_tune_nibi.sh config/config_alliance_lean.yaml --grid config/my_grid.yaml
#   sbatch jobs/step3b_tune_nibi.sh config/config_alliance_lean.yaml --full-refresh
#
# Then: step3c (calibration) and step 3 with the tuned config, e.g.
#   sbatch jobs/step3c_calibrate_nibi.sh config/config_alliance_lean_tuned.yaml
#   sbatch jobs/step3_run_experiment_nibi.sh config/config_alliance_lean_tuned.yaml
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base.
# =============================================================================
#SBATCH --job-name=mimic-twin-step3b-tune
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"
CONFIG="${1:-config/config_alliance_lean.yaml}"
echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) — config=$CONFIG =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/ollama_lib.sh
start_ollama "$CONFIG"         # OLLAMA_NUM_PARALLEL / FLASH_ATTENTION from the config
require_models "$CONFIG" default     # tuning uses the primary model only

python src/tune.py --config-file "$CONFIG" "${@:2}"
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
