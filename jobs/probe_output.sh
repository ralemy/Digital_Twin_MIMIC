#!/bin/bash
# =============================================================================
# Slurm batch job — what one LLM variant actually generates for the
# forecasting prompt, with the JSON-schema format the pipeline now sends and
# with plain format=json as before, and under the schema with reasoning
# off (think=false) — src/probe_output.py. Diagnostic only:
# nothing is checkpointed.
#
#   cd "$DT_REPO"     # after the setup (README, section 1)
#   bash jobs/submit.sh jobs/probe_output.sh config/config_alliance_lean_tuned.yaml baichuan_m2
#   bash jobs/submit.sh jobs/probe_output.sh <config> <variant|default> --n-patients 2 --max-tokens 4096
#
# Arguments: the config, the llm variant (default baichuan_m2; 'default' for
# llm.model), then any of probe_output.py's options. Requests are sent one at
# a time with a 4096-token budget, so each answer can end on its own.
#
# Output: per call, a log line with why generation stopped, tokens, whether
# the answer is usable, the longest decimals and the start/end of the text;
# a summary table at the end; full outputs in
# <results_dir>/probe_output/probe-<job id>.jsonl (patient-derived model
# output: same custody as the other results).
#
# Time: ~30 s-3 min to load a 32B model, then up to ~6 min per call that
# runs to 4096 tokens; the default 4 patients x 3 modes fits in the 1 h limit.
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base.
# =============================================================================
#SBATCH --job-name=mimic-twin-probe-output
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; jobs/submit.sh overrides it

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO"
CONFIG="${1:-config/config_alliance_lean_tuned.yaml}"
VARIANT="${2:-baichuan_m2}"
echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) — config=$CONFIG variant=$VARIANT =="
[ -f "$CONFIG" ] || { echo "== $CONFIG not found — pass a config ==" >&2; exit 1; }

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/ollama_lib.sh
start_ollama "$CONFIG"
require_models "$CONFIG" all

python src/probe_output.py --config-file "$CONFIG" --variant "$VARIANT" "${@:3}"
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
