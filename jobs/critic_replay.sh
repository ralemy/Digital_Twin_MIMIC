#!/bin/bash
# =============================================================================
# Slurm batch job — RQ2 with the critic before post-processing, replayed on a
# finished run's saved forecasts (src/critic_replay.py). Only the LLM critic
# arm needs the GPU: the condition's critic reviews the forecasts that have
# out-of-range values (one short request per such variable). Its corrections
# are cached in <results_dir>/critic_replay/, so later re-runs with other
# options need no GPU (python src/critic_replay.py ... on a login node).
#
#   cd "$DT_REPO"     # after the setup (README, section 1)
#   bash jobs/submit.sh jobs/critic_replay.sh tuned_configs/<run>/<config>_tuned.yaml [critic_replay.py options]
#
# Time: ~1-10 min to read and load Med42-70B, then a few hundred requests at
# llm_max_concurrent_requests at once (nibi_full_oct13: 124 + 106 patients
# flagged); well inside 1 h. Memory as step 3 (the Med42 runner was
# OOM-killed at 64000M on Fir).
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base.
# =============================================================================
#SBATCH --job-name=mimic-twin-critic-replay
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; jobs/submit.sh overrides it

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"
CONFIG="${1:?pass the tuned config of the finished run}"
echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) — config=$CONFIG =="
[ -f "$CONFIG" ] || { echo "== $CONFIG not found ==" >&2; exit 1; }

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/ollama_lib.sh
start_ollama "$CONFIG"
require_models "$CONFIG" all

python src/critic_replay.py "$CONFIG" "${@:2}"
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
