#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 2/4: src/extract_cohort.py
# Builds the analytic cohort and extracts the variable panel with DuckDB,
# running directly over the raw chartevents/labevents .csv(.gz) files —
# this is the CPU/RAM-heavy step (matches the config's
# performance.duckdb_threads / duckdb_memory_limit_gb). No GPU, no Ollama.
# Requires step 1 (item_mapping.json) to already exist.
#
# Settings (project directory, data_root) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments. Submit from the project directory.
#
# Works for either scope — pass the config file as the first argument:
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
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
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step2-extract
#SBATCH --cpus-per-task=12
#SBATCH --mem=192000M
#SBATCH --time=02:00:00
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
