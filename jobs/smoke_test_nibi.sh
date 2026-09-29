#!/bin/bash
# =============================================================================
# Slurm batch job — smoke test only. Run this BEFORE any real-data job to
# confirm the codebase and your venv actually work on Nibi. No GPU, no
# Ollama, no MIMIC-IV data needed — src/smoke_test.py fabricates a small
# synthetic cohort and mocks the LLM (see its own docstring). Finishes in
# well under a minute of actual work; the time/resource limits below are
# generous headroom, not an estimate of what it needs.
#
# Reads the project directory from $AGENTIC_DT_PRJ (must be exported in the
# submitting shell — sbatch passes the submission environment through by
# default, so `export AGENTIC_DT_PRJ=...` before `sbatch` is enough).
#
# Submit with:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/smoke_test_nibi.sh
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${AGENTIC_DT_PRJ:?AGENTIC_DT_PRJ is not set — export it to the project directory before sbatch, e.g. export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1}"
PROJECT_DIR="$AGENTIC_DT_PRJ"
cd "$PROJECT_DIR"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  project_dir=$PROJECT_DIR =="

module load python/3.11
source "$PROJECT_DIR/.venv/bin/activate"

python src/smoke_test.py

echo "== job $SLURM_JOB_ID finished at $(date) =="
