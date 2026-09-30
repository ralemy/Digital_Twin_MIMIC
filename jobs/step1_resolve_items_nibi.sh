#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 1/4: src/resolve_items.py
# Resolves the variable panel to real MIMIC-IV itemids by querying
# icu/d_items and hosp/d_labitems (small dictionary tables — this step is
# quick and does NOT touch chartevents/labevents themselves). No GPU, no
# Ollama needed.
#
# Reads the project directory from $AGENTIC_DT_PRJ (export it in the
# submitting shell before `sbatch` — sbatch passes the submission
# environment through by default).
#
# Works for either scope — pass the config file as the first argument:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/step1_resolve_items_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step1_resolve_items_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
#
# Writes <cache_dir>/item_mapping.json — open and read it before running
# step 2, exactly as the README instructs.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step1-resolve
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

echo "== step 1/4: resolve_items.py =="
python src/resolve_items.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
