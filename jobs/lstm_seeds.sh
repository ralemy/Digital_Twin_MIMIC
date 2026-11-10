#!/bin/bash
# =============================================================================
# Slurm batch job — src/lstm_seeds.py: the LSTM baseline re-trained with
# other seeds on finished runs' patients (no LLM, no Ollama; the GPU only
# speeds up training, a few minutes per seed and run). One job takes every
# run given, so the short trainings are bundled.
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the
# repository base, with the runs' tuned configs:
#   cd "$DT_REPO"
#   bash jobs/submit.sh jobs/lstm_seeds.sh \
#       tuned_configs/tri_lean_exp4/config_alliance_lean_exp4_tuned.yaml [more tuned configs] [--seeds 42,43,44,45,46]
#
# Writes <results_dir>/lstm_seeds/lstm_seed<N>_raw.npz for each run and seed;
# seeds already saved are skipped, so a resubmitted job carries on.
# =============================================================================
#SBATCH --job-name=mimic-twin-lstm-seeds
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"
[ "$#" -gt 0 ] || { echo "== give at least one finished run's tuned config ==" >&2; exit 1; }
echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) — $* =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"

python src/lstm_seeds.py "$@"
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
