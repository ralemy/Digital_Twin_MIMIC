#!/bin/bash
# =============================================================================
# Submit one of the job scripts, adapted to this cluster — run on a LOGIN NODE:
#   bash jobs/submit.sh [sbatch options] jobs/<job>.sh [job arguments]
# e.g.
#   bash jobs/submit.sh jobs/step1_resolve_items_nibi.sh config/config_alliance_lean.yaml
#   bash jobs/submit.sh --time=02:00:00 jobs/step3b_tune_nibi.sh config/config_alliance_lean.yaml
# sbatch options go before the script, in --option=value form. jobs/run_all.sh
# submits every job through this script.
#
# What it adds to plain `sbatch jobs/<job>.sh`:
#   - submits from the repo base (the jobs find jobs/load_profile.sh there)
#   - --output=$DT_LOG_DIR/%x-%j.out unless an --output is given, so job
#     output goes to paths.logs_dir of your profile (on $SCRATCH where compute
#     nodes can't write the repo, e.g. Trillium)
#   - with slurm.gpu_jobs_only (the default on Trillium, whose GPU subcluster
#     takes only jobs with a GPU and no memory request): submits a copy of the
#     script whose #SBATCH lines ask for --nodes=1 --gpus-per-node=1 and no
#     --mem (a quarter node: 24 cores, ~188 GiB, 1 H100). The CPU-only steps
#     (resolve, extract, evaluate) then also hold a GPU for their few minutes.
# Prints what sbatch prints (the job id with --parsable).
# =============================================================================
set -euo pipefail

SBATCH_OPTS=()
while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do
    SBATCH_OPTS+=("$1"); shift
done
if [ $# -eq 0 ] || [ ! -f "$1" ]; then
    echo "usage: bash jobs/submit.sh [--sbatch-option=value ...] jobs/<job>.sh [job arguments]" >&2
    exit 1
fi
JOB_SCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift

# The job's own --profile <file>, if any, is the profile to adapt to.
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --profile) DT_PROFILE="${args[i + 1]:-}" ;;
        --profile=*) DT_PROFILE="${args[i]#--profile=}" ;;
    esac
done
export DT_PROFILE
# shellcheck source=jobs/setup_bash.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_bash.sh" || exit 1
cd "$DT_REPO"

has_output=0
for opt in "${SBATCH_OPTS[@]}"; do
    case "$opt" in --output=*|-o*) has_output=1 ;; esac
done
if [ "$has_output" -eq 0 ]; then
    mkdir -p "$DT_LOG_DIR"
    SBATCH_OPTS+=("--output=$DT_LOG_DIR/%x-%j.out")
fi

if [ "$DT_GPU_JOBS_ONLY" = 1 ]; then
    if [ "$DT_CLUSTER" = trillium ]; then
        case "$(hostname -s)" in
            trig-login*) ;;
            *) echo "== on Trillium, submit from the GPU login node (ssh trillium-gpu.scinet.utoronto.ca, or ssh trig-login01 from here) ==" >&2
               exit 1 ;;
        esac
    fi
    kept=()
    for opt in "${SBATCH_OPTS[@]}"; do
        case "$opt" in --mem=*|--mem-per-*|--gpus-per-node=*|--gres=*|--gpus=*) ;; *) kept+=("$opt") ;; esac
    done
    SBATCH_OPTS=("${kept[@]}")
    adapted=$(mktemp "${TMPDIR:-/tmp}/dt-submit-$(basename "$JOB_SCRIPT" .sh)-XXXXXX.sh")
    trap 'rm -f "$adapted"' EXIT
    # Drop the memory and GPU requests; ask for one GPU right after the shebang.
    awk 'NR == 1 { print; print "#SBATCH --nodes=1"; print "#SBATCH --gpus-per-node=1"; next }
         /^#SBATCH[[:space:]]+--(mem|mem-per-cpu|mem-per-gpu|gpus-per-node|gres|gpus)([=[:space:]]|$)/ { next }
         { print }' "$JOB_SCRIPT" > "$adapted"
    JOB_SCRIPT=$adapted
fi

sbatch "${SBATCH_OPTS[@]}" "$JOB_SCRIPT" "$@"
