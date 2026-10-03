#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 1/4: src/resolve_items.py
# Resolves the variable panel to real MIMIC-IV itemids by querying
# icu/d_items and hosp/d_labitems (small dictionary tables — this step is
# quick and does NOT touch chartevents/labevents themselves). No GPU, no
# Ollama needed.
#
# Settings (project directory, data_root) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments. Submit from the project directory.
#
# Works for either scope — pass the config file as the first argument:
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
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
#SBATCH --mem=32000M
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
echo "starting"
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
