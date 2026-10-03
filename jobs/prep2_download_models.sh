#!/bin/bash
# =============================================================================
# Slurm batch job — download every Ollama model a config's run needs (the
# default llm.model plus each llm.variants entry that `conditions` uses),
# and create its `alias` (`ollama cp <model> <alias>`) where the config sets
# one. Run this before step 3, which can't pull models mid-job.
#
# For each model it checks what is already in $OLLAMA_MODELS and only does
# what's missing:
#   - alias set and present (same weights as model)  -> nothing to do
#   - model present, alias missing                   -> ollama cp only
#   - model missing                                  -> ollama pull (+ cp)
#   - alias present but pointing at different
#     weights than model (e.g. model re-pulled)       -> ollama cp again
# Re-submitting after a timeout or failure continues where it left off
# (`ollama pull` resumes partial downloads).
#
# All Nibi nodes have internet access (Alliance docs, Nibi > Site specifics),
# so this runs as a regular CPU-only job — no GPU needed to download.
# Single-core and network bound.
#
# Settings (project directory, Ollama model store — the same one step 3 reads) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments.
# Check free space first — a 70B Q4 model alone is ~43GB — then submit from
# the project directory:
#   diskusage_report
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/prep2_download_models.sh                                # config/config_nibi_lean.yaml
#   sbatch jobs/prep2_download_models.sh config/config_nibi_full_variables.yaml
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-prep2-models
#SBATCH --cpus-per-task=1
# --mem: job 23132444 peaked at 3.88 of 3.91 GB while ollama pulled a 27B model
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Settings come from the profile (~/.config/dt_profile.yml, or --profile
# <file> among this job's arguments) — see jobs/load_profile.sh. The
# remaining arguments are this job's own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the project directory: cd <project> && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

command -v ollama > /dev/null || { echo "== ollama not found in \$HOME/ollama-local/bin (paths.ollama_bin in your profile) — install it per installing_ollama.md ==" >&2; exit 1; }
PROJECT_DIR="$AGENTIC_DT_PRJ"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config_nibi_lean.yaml}"
[ -f "$CONFIG" ] || { echo "== config '$CONFIG' not found under $PROJECT_DIR ==" >&2; exit 1; }

# A compute node can be shared with other users' jobs, which may run their
# own ollama on the default 11434 — use a per-job port so we never talk to
# someone else's server (or pull into their model store).
OLLAMA_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"   # read by both `ollama serve` and the client commands
export OLLAMA_MODELS                               # already set by the submitting shell (checked above)

echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  config=$CONFIG  OLLAMA_MODELS=$OLLAMA_MODELS =="

module load python/3.11
source "$PROJECT_DIR/.venv/bin/activate"

# One "model<TAB>alias" line per model the config's run needs (alias empty
# if none is set).
SPECS=$(python -c 'import sys; sys.path.insert(0, "src"); from common import load_config, llm_models_to_set_up; [print(m, a or "", sep="\t") for m, a in llm_models_to_set_up(load_config(sys.argv[1]))]' "$CONFIG")
if [ -z "$SPECS" ]; then
    echo "== config uses no LLM conditions — nothing to download =="
    exit 0
fi
echo "== models needed by $CONFIG (model -> alias) =="
printf '%s\n' "$SPECS" | awk -F'\t' '{print "   " $1 ($2 ? "  ->  " $2 : "")}'

# --- start Ollama in the background, on this node only ---------------------
mkdir -p "$OLLAMA_MODELS"
OLLAMA_LOG="ollama-prep2-${SLURM_JOB_ID:-local}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    echo "== stopping ollama (pid $OLLAMA_PID) at $(date) =="
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "== waiting for ollama on ${OLLAMA_HOST} =="
if ! curl -sf --retry 60 --retry-delay 1 --retry-connrefused "http://${OLLAMA_HOST}/api/tags" > /dev/null; then
    echo "== ollama did not become ready — see $OLLAMA_LOG. Aborting. ==" >&2
    exit 1
fi
echo "== ollama ready =="

# The ID column of `ollama list` for a name, empty if not present. `ollama
# show` resolves names the way Ollama does (e.g. adds ':latest'), so use it
# for the presence test and `ollama list` only to read the ID.
model_id() {
    ollama show "$1" > /dev/null 2>&1 || return 0
    local name="$1"
    [[ "$name" == *:* ]] || name="$name:latest"
    ollama list | awk -v n="$name" 'NR > 1 && tolower($1) == tolower(n) && !found {print $2; found = 1}'
}

FAILED=()
while IFS=$'\t' read -r MODEL ALIAS; do
    echo
    echo "== $MODEL${ALIAS:+  (alias $ALIAS)} =="
    MODEL_ID=$(model_id "$MODEL")
    ALIAS_ID=""
    [ -n "$ALIAS" ] && ALIAS_ID=$(model_id "$ALIAS")

    if [ -n "$ALIAS" ] && [ -n "$ALIAS_ID" ] && { [ -z "$MODEL_ID" ] || [ "$ALIAS_ID" = "$MODEL_ID" ]; }; then
        echo "   alias '$ALIAS' already present ($ALIAS_ID) — nothing to do"
        continue
    fi
    if [ -z "$ALIAS" ] && [ -n "$MODEL_ID" ]; then
        echo "   already present ($MODEL_ID) — nothing to do"
        continue
    fi

    if [ -z "$MODEL_ID" ]; then
        echo "   not present — pulling (started $(date +%T))"
        # The progress bar is terminal cursor codes, unreadable in a log file:
        # strip the codes and progress lines, keep the last few status lines.
        # Success is judged by model_id below, not by this pipeline's status.
        ollama pull "$MODEL" 2>&1 | tr '\r' '\n' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
            | grep -v -E '^\s*$|[0-9]+%' | uniq | tail -5 || true
        MODEL_ID=$(model_id "$MODEL")
        if [ -z "$MODEL_ID" ]; then
            echo "   PULL FAILED for '$MODEL' — see above and $OLLAMA_LOG" >&2
            FAILED+=("$MODEL")
            continue
        fi
        echo "   pulled ($MODEL_ID) at $(date +%T)"
    else
        echo "   model present ($MODEL_ID)"
    fi

    if [ -n "$ALIAS" ]; then
        [ -n "$ALIAS_ID" ] && echo "   alias '$ALIAS' points at different weights ($ALIAS_ID) — re-creating it"
        if ollama cp "$MODEL" "$ALIAS"; then
            echo "   alias '$ALIAS' -> '$MODEL' created"
        else
            echo "   ALIAS FAILED: ollama cp $MODEL $ALIAS" >&2
            FAILED+=("$ALIAS")
        fi
    fi
done <<< "$SPECS"

echo
echo "== ollama list =="
ollama list

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "== job finished with failures: ${FAILED[*]} — re-submit to retry (downloads resume) ==" >&2
    exit 1
fi
echo "== all models for $CONFIG are ready in $OLLAMA_MODELS — job finished at $(date) =="
# `trap cleanup EXIT` stops ollama on the way out, success or failure.
