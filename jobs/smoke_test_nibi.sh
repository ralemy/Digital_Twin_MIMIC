#!/bin/bash
# =============================================================================
# Slurm batch job — smoke test only. Run this BEFORE any real-data job to
# confirm the codebase and your venv actually work on Nibi. No GPU, no
# Ollama, no MIMIC-IV data needed — src/smoke_test.py fabricates a small
# synthetic cohort and mocks the LLM (see its own docstring). Finishes in
# well under a minute of actual work; the time/resource limits below are
# generous headroom, not an estimate of what it needs.
#
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   sbatch jobs/smoke_test_nibi.sh
# =============================================================================
#SBATCH --job-name=mimic-twin-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

# Settings come from your profile, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  repo=$DT_REPO =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"

python src/smoke_test.py --config-file config/config_alliance_lean.yaml

echo "== job $SLURM_JOB_ID finished at $(date) =="
