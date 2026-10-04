#!/bin/bash
# =============================================================================
# Slurm batch job — pipeline step 4/4: src/evaluate_results.py
# RQ1/RQ2/RQ3 statistical analysis (paired Wilcoxon tests, bootstrap 95% CIs
# resampling patients) over step 3's *_raw.npz output. Pure CPU/NumPy/SciPy
# work — no GPU, no Ollama. Requires step 3 to already be done.
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base:
#
# Works for either scope — pass the config file as the first argument:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   sbatch jobs/step4_evaluate_results_nibi.sh config/config_alliance_lean.yaml
#   sbatch jobs/step4_evaluate_results_nibi.sh config/config_alliance_full.yaml
# Defaults to config_alliance_lean.yaml if omitted.
#
# Writes <results_dir>/statistical_analysis.json.
# =============================================================================
#SBATCH --job-name=mimic-twin-step4-evaluate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

# Settings come from your profile, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"

CONFIG="${1:-config/config_alliance_lean.yaml}"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  repo=$DT_REPO  config=$CONFIG =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"

echo "== step 4/4: evaluate_results.py =="
python src/evaluate_results.py --config-file "$CONFIG"

echo "== job $SLURM_JOB_ID finished at $(date) =="
