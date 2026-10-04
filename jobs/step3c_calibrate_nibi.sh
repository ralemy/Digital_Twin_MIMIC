#!/bin/bash
# =============================================================================
# Slurm batch job — split-conformal interval calibration (src/calibrate.py)
# on the CALIBRATION subset of the validation split (the validation stays not
# used for tuning). Requires steps 1 and 2; normally run with the tuned config
# written by step3b. Never reads the test split.
#
# Every condition in the config forecasts the calibration patients, and
# per-variable interval scaling factors are written to
# <results_dir>/calibration.json; step 4 (evaluate_results.py) applies them to
# the test forecasts and reports coverage/width before and after.
# Cost: like step 3 on 322 instead of 450 patients (~70% of its time).
#
# Resuming: forecasts are checkpointed per batch; after a time limit or
# failure submit it again, or chain a continuation:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   JOB=$(sbatch --parsable jobs/step3c_calibrate_nibi.sh config/config_alliance_lean_tuned.yaml)
#   sbatch --dependency=afterany:$JOB jobs/step3c_calibrate_nibi.sh config/config_alliance_lean_tuned.yaml
# Other arguments after the config go to calibrate.py (e.g. --full-refresh).
# Defaults to config_alliance_lean.yaml if omitted; pass the tuned config
# (config_alliance_lean_tuned.yaml, written by step3b) for the normal flow.
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base.
# =============================================================================
#SBATCH --job-name=mimic-twin-step3c-calibrate
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
[ -f "$CONFIG" ] || { echo "== $CONFIG not found — run step3b (tuning) first, or pass a config ==" >&2; exit 1; }

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/ollama_lib.sh
start_ollama "$CONFIG"         # OLLAMA_NUM_PARALLEL / FLASH_ATTENTION from the config
require_models "$CONFIG" all

python src/calibrate.py --config-file "$CONFIG" "${@:2}"
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
