#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 2/4: src/extract_cohort.py
# Builds the analytic cohort and extracts the variable panel with DuckDB,
# running directly over the raw chartevents/labevents .csv(.gz) files —
# this is the CPU/RAM-heavy step (matches the config's
# performance.duckdb_threads / duckdb_memory_limit_gb). No GPU, no Ollama.
# Requires step 1 (item_mapping.json) to already exist.
#
# Paths (repo, MIMIC-IV, Ollama models, modules) come from
# jobs/setup_bash.sh, picked by cluster. Submit from the repository base:
#
# Works for either scope — pass the config file as the first argument:
#   cd "$DT_REPO"     # with jobs/setup_bash.sh sourced (sets DT_REPO, SBATCH_ACCOUNT)
#   sbatch jobs/step2_extract_cohort_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step2_extract_cohort_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
#
# --cpus-per-task/--mem below pair with config_nibi_*.yaml's duckdb_threads=10
# / duckdb_memory_limit_gb=120. --mem=192000M (~201 GB) must stay well above
# the DuckDB limit: after DuckDB finishes, the script loads the cached panel
# into pandas, and that needs its own headroom (~80 GB here). DuckDB spills
# to the node-local $SLURM_TMPDIR (3 TB on Nibi CPU nodes) past its limit.
# If you change one, change the other; check actual peak usage afterwards
# with `seff <jobid>` and trim --mem if it was far below.
# =============================================================================
#SBATCH --job-name=mimic-twin-step2-extract
#SBATCH --cpus-per-task=12
#SBATCH --mem=192000M
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

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

echo "== step 2/4: extract_cohort.py =="
python src/extract_cohort.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
