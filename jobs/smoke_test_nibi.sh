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
# Settings (project directory) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments. Submit from the project directory:
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/smoke_test_nibi.sh
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
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

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  project_dir=$PROJECT_DIR =="

module load python/3.11
source "$PROJECT_DIR/.venv/bin/activate"

python src/smoke_test.py --config-file config/config_nibi_lean.yaml

echo "== job $SLURM_JOB_ID finished at $(date) =="
