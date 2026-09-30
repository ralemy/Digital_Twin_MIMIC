#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 4/4: src/evaluate_results.py
# RQ1/RQ2/RQ3 statistical analysis (paired Wilcoxon tests, bootstrap 95% CIs
# resampling patients) over step 3's *_raw.npz output. Pure CPU/NumPy/SciPy
# work — no GPU, no Ollama. Requires step 3 to already be done.
#
# Reads the project directory from $AGENTIC_DT_PRJ (export it in the
# submitting shell before `sbatch` — sbatch passes the submission
# environment through by default).
#
# Works for either scope — pass the config file as the first argument:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/step4_evaluate_results_nibi.sh config/config_nibi_lean.yaml
#   sbatch jobs/step4_evaluate_results_nibi.sh config/config_nibi_full_variables.yaml
# Defaults to config_nibi_lean.yaml (the HREB-approved scope) if omitted.
#
# Writes <results_dir>/statistical_analysis.json.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-step4-evaluate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
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

echo "== step 4/4: evaluate_results.py =="
python src/evaluate_results.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
