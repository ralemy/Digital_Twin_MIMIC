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
# The Alliance configs use six models (docs/llm_selection.docx):
#   qwen2.5:32b-instruct-q8_0 (alias qwen2.5:32b)            ~35 GB  Ollama library
#   Baichuan-M2-32B Q8_0      (alias baichuan-m2:32b)        ~35 GB  Hugging Face (bartowski)
#   gemma3:27b                                               ~17 GB  Ollama library
#   MedGemma 27B text Q4_K_M  (alias medgemma:27b)           ~17 GB  Hugging Face (unsloth)
#   llama3:70b-instruct-q4_K_M                               ~43 GB  Ollama library
#   Llama3-Med42-70B Q4_K_M   (alias med42:70b)              ~43 GB  Hugging Face (mradermacher)
# None of these downloads needs a login or a licence click-through: the
# Ollama library models and the three Hugging Face GGUF repositories are
# ungated. Their licences still apply to how you use them (Gemma Terms of
# Use, Health AI Developer Foundations terms for MedGemma, Meta Llama 3
# Community Licence and Acceptable Use Policy for Llama 3 and Med42,
# Apache 2.0 for Qwen2.5 and Baichuan-M2). jobs/run_all.sh runs this job as
# its first stage.
#
# CPU-only, single-core and network bound — no GPU needed to download. Where
# worker nodes have internet access (environment.workers_have_internet: true),
# submit it with jobs/submit.sh. Where they don't (false — e.g. Rorqual,
# Trillium), run it on a login node instead, inside tmux:
#   cd "$DT_REPO" && bash jobs/prep2_download_models.sh <config>
# As a Slurm job on such a cluster it stops at once with that advice;
# jobs/run_all.sh's models stage runs it on the login node by itself.
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh.
# Check free space first — a 70B Q4 model alone is ~43GB — then submit from
# the repository base:
#   diskusage_report
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   bash jobs/submit.sh jobs/prep2_download_models.sh                                # config/config_alliance_lean.yaml
#   bash jobs/submit.sh jobs/prep2_download_models.sh config/config_alliance_full.yaml
# =============================================================================
#SBATCH --job-name=mimic-twin-prep2-models
#SBATCH --cpus-per-task=1
# --mem: job 23132444 peaked at 3.88 of 3.91 GB while ollama pulled a 27B model
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

# Settings come from your profile, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

# Downloading needs the internet. On clusters whose worker nodes have none
# (environment.workers_have_internet: false in your profile), run this on a
# login node with bash instead of as a job; jobs/run_all.sh does that itself.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$DT_WORKER_INTERNET" != 1 ]; then
    echo "== this cluster's worker nodes have no internet access (environment.workers_have_internet: false in your profile) — run it on a login node instead, e.g. inside tmux: cd \$DT_REPO && bash jobs/prep2_download_models.sh $* ==" >&2
    exit 1
fi

command -v ollama > /dev/null || { echo "== ollama not found in $DT_OLLAMA_BIN (paths.ollama_bin in your profile) — the setup installs it (README, section 2) ==" >&2; exit 1; }
cd "$DT_REPO"

CONFIG="${1:-config/config_alliance_lean.yaml}"
[ -f "$CONFIG" ] || { echo "== config '$CONFIG' not found under $DT_REPO ==" >&2; exit 1; }

# A compute node can be shared with other users' jobs, which may run their
# own ollama on the default 11434 — use a per-job port so we never talk to
# someone else's server (or pull into their model store).
OLLAMA_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"   # read by both `ollama serve` and the client commands
export OLLAMA_MODELS                               # already set by the submitting shell (checked above)

echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  config=$CONFIG  OLLAMA_MODELS=$OLLAMA_MODELS =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/progress_lib.sh

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
mkdir -p "$DT_LOG_DIR"
OLLAMA_LOG="$DT_LOG_DIR/ollama-prep2-${SLURM_JOB_ID:-local}.log"
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
        # Progress: a live bar in a terminal; in a log, a line per 10% or per
        # minute (jobs/progress_lib.sh). Success is judged by model_id below,
        # not by this pipeline's status.
        ollama pull "$MODEL" 2>&1 | dt_progress "$MODEL" || true
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
