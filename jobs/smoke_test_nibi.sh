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
# Paths (repo, MIMIC-IV, Ollama models, modules) come from
# jobs/setup_bash.sh, picked by cluster. Submit from the repository base:
#   cd "$DT_REPO"     # with jobs/setup_bash.sh sourced (sets DT_REPO, SBATCH_ACCOUNT)
#   sbatch jobs/smoke_test_nibi.sh
# =============================================================================
#SBATCH --job-name=mimic-twin-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Paths come from jobs/setup_bash.sh, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  repo=$DT_REPO =="

module load $DT_MODULES          # jobs/setup_bash.sh
source "$DT_REPO/.venv/bin/activate"

python src/smoke_test.py --config-file config/config_nibi_lean.yaml

echo "== job $SLURM_JOB_ID finished at $(date) =="
