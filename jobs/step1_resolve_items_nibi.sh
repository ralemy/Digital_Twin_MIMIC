#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 1/4: src/resolve_items.py
# Resolves the variable panel to real MIMIC-IV itemids by querying
# icu/d_items and hosp/d_labitems (small dictionary tables — this step is
# quick and does NOT touch chartevents/labevents themselves). No GPU, no
# Ollama needed.
#
# Paths (repo, MIMIC-IV, Ollama models, modules) come from
# jobs/setup_bash.sh, picked by cluster. Submit from the repository base:
#
# Works for either scope — pass the config file as the first argument:
#   cd "$DT_REPO"     # with jobs/setup_bash.sh sourced (sets DT_REPO, SBATCH_ACCOUNT)
#   sbatch jobs/step1_resolve_items_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step1_resolve_items_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
#
# Writes <cache_dir>/item_mapping.json — open and read it before running
# step 2, exactly as the README instructs.
# =============================================================================
#SBATCH --job-name=mimic-twin-step1-resolve
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:15:00
#SBATCH --output=%x-%j.out
echo "starting"
set -euo pipefail

# Paths come from jobs/setup_bash.sh, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"

CONFIG="${1:-config/config_nibi_lean.yaml}"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  repo=$DT_REPO  config=$CONFIG =="

if [[ "$CONFIG" == *full_variables* ]]; then
    echo "== NOTE: this is the 19-variable ALTERNATE SCOPE, not covered by"
    echo "== current UVic HREB approval — see config/config_nibi_full_variables.yaml"
fi

module load $DT_MODULES          # jobs/setup_bash.sh
source "$DT_REPO/.venv/bin/activate"

echo "== step 1/4: resolve_items.py =="
python src/resolve_items.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
