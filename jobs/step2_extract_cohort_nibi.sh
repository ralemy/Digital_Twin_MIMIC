#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 2/4: src/extract_cohort.py
# Builds the analytic cohort and extracts the variable panel with DuckDB,
# running directly over the raw chartevents/labevents .csv(.gz) files —
# this is the CPU/RAM-heavy step (matches the config's
# performance.duckdb_threads / duckdb_memory_limit_gb). No GPU, no Ollama.
# Requires step 1 (item_mapping.json) to already exist.
#
# Reads the project directory from $AGENTIC_DT_PRJ (export it in the
# submitting shell before `sbatch` — sbatch passes the submission
# environment through by default).
#
# Works for either scope — pass the config file as the first argument:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/step2_extract_cohort_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step2_extract_cohort_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
#
# --cpus-per-task/--mem below match config_nibi_*.yaml's duckdb_threads=10 /
# duckdb_memory_limit_gb=200 with a little headroom on cores; those config
# numbers are themselves conservative estimates (Nibi's own per-node spec
# sheet wasn't reachable while preparing this) — check
# `sinfo -o "%N %c %m %G"` and raise both together if more is available.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step2-extract
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${AGENTIC_DT_PRJ:?AGENTIC_DT_PRJ is not set — export it to the project directory before sbatch, e.g. export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1}"
PROJECT_DIR="$AGENTIC_DT_PRJ"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config_nibi_lean.yaml}"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  project_dir=$PROJECT_DIR  config=$CONFIG =="

if [[ "$CONFIG" == *full_variables* ]]; then
    echo "== NOTE: this is the 19-variable ALTERNATE SCOPE, not covered by"
    echo "== current UVic HREB approval — see config/config_nibi_full_variables.yaml"
fi

module load python/3.11
source "$PROJECT_DIR/.venv/bin/activate"

echo "== step 2/4: extract_cohort.py =="
python src/extract_cohort.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
