#!/bin/bash
# =============================================================================
# Pack one jobs/run_all.sh run into <run_name>_results.tar — run on a LOGIN
# NODE (it only reads and archives files):
#   bash jobs/pack_run.sh <run_name> [-o <output dir>] [--include-checkpoints] [--profile <file>]
#
# The archive holds, for whatever exists of the run:
#   logs/                       $DT_LOGS_ROOT/<run>/ (driver log, job .out
#                               files, Ollama logs), plus the run's Ollama
#                               logs found directly in $DT_LOGS_ROOT (by the
#                               job ids in run_all/<run>/*.state): jobs before
#                               run_all.sh passed --log-dir wrote them there
#                               on Trillium
#   mimic-iv-twin-work/         $DT_RESULTS_DIR/mimic-iv-twin-work/<run>/ (lean)
#   mimic-iv-twin-work-full/    $DT_RESULTS_DIR/mimic-iv-twin-work-full/<run>/ (full)
#   tuned_configs/              tuned_configs/<run>/ (the run's config and tuned config)
# Options:
#   -o, --output <dir>   where to write the tar (default: the current directory)
#   --include-checkpoints  also pack checkpoints/ and checkpoints_tuned/, left
#                        out by default (resume state only; the results don't
#                        need them, and they are a large share of the size)
#   --profile <file>     the profile to read the locations from (as in run_all.sh)
#
# DATA USE: the work directories hold patient-level data derived from
# MIMIC-IV (cohort.parquet, panel_long.parquet, cache/, *_raw.npz). The tar
# may only be kept or copied where the MIMIC-IV Data Use Agreement allows —
# e.g. your project space, not a laptop or a shared drive.
#
# The result is an uncompressed tar: the bulk of it (parquet, npz) is
# compressed already. Copy it to project space with Globus or rsync.
# =============================================================================
set -uo pipefail

usage() { sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

RUN_NAME=""; OUT_DIR="$PWD"; INCLUDE_CHECKPOINTS=0; PROFILE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output) OUT_DIR="${2:-}"; shift 2 ;;
        --output=*) OUT_DIR="${1#--output=}"; shift ;;
        --include-checkpoints) INCLUDE_CHECKPOINTS=1; shift ;;
        --profile) PROFILE_ARGS+=("$1" "${2:-}"); shift 2 ;;
        --profile=*) PROFILE_ARGS+=("$1"); shift ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown option: $1" >&2; usage 1 ;;
        *) [ -z "$RUN_NAME" ] || { echo "== one run name only ==" >&2; exit 1; }
           RUN_NAME=$1; shift ;;
    esac
done
[ -n "$RUN_NAME" ] || usage 1
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "== run name: letters, digits, '_', '.' and '-' only (got '$RUN_NAME') ==" >&2
    exit 1
fi
OUT_DIR=$(cd "$OUT_DIR" 2>/dev/null && pwd) || { echo "== output directory not found ==" >&2; exit 1; }

source "$(cd "$(dirname "$0")" && pwd)/load_profile.sh" || exit 1
load_profile "${PROFILE_ARGS[@]}" || exit 1
cd "$DT_REPO" || exit 1

# <name in the archive>:<directory of this run> (logs/ is assembled below)
PARTS=(
    "mimic-iv-twin-work:$DT_RESULTS_DIR/mimic-iv-twin-work/$RUN_NAME"
    "mimic-iv-twin-work-full:$DT_RESULTS_DIR/mimic-iv-twin-work-full/$RUN_NAME"
    "tuned_configs:$DT_REPO/tuned_configs/$RUN_NAME"
)

# A staging directory of links named as in the archive; tar --dereference
# stores what they point to under those names.
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pack_run.XXXXXX") || exit 1
cleanup() { find "$STAGE" -mindepth 1 -type l -delete; rmdir "$STAGE/logs" 2>/dev/null; rmdir "$STAGE"; }
trap cleanup EXIT

found=()

# logs/: everything in the run's log folder, plus its jobs' Ollama logs that
# landed directly in $DT_LOGS_ROOT.
mkdir "$STAGE/logs"
n_logs=0
if [ -d "$DT_LOGS_ROOT/$RUN_NAME" ]; then
    for f in "$DT_LOGS_ROOT/$RUN_NAME"/*; do
        [ -e "$f" ] && ln -s "$f" "$STAGE/logs/" && n_logs=$((n_logs + 1))
    done
fi
n_extra=0
for id in $(awk '{for (i = 4; i <= NF; i++) if ($i ~ /^[0-9]+$/) print $i}' run_all/"$RUN_NAME"/*.state 2>/dev/null | sort -u); do
    for f in "$DT_LOGS_ROOT"/ollama-"$id".log "$DT_LOGS_ROOT"/ollama-"$id"-*.log; do
        [ -e "$f" ] && [ ! -e "$STAGE/logs/$(basename "$f")" ] && ln -s "$f" "$STAGE/logs/" && n_extra=$((n_extra + 1))
    done
done
if [ $((n_logs + n_extra)) -gt 0 ]; then
    found+=(logs)
    printf '   %-24s <- %s (%d files) + %d Ollama log(s) from %s\n' "logs/" "$DT_LOGS_ROOT/$RUN_NAME" \
        "$n_logs" "$n_extra" "$DT_LOGS_ROOT"
else
    printf '   %-24s    not found: %s\n' "logs/" "$DT_LOGS_ROOT/$RUN_NAME"
fi

for part in "${PARTS[@]}"; do
    name=${part%%:*}; dir=${part#*:}
    if [ -d "$dir" ]; then
        ln -s "$dir" "$STAGE/$name"
        found+=("$name")
        printf '   %-24s <- %s (%s)\n' "$name/" "$dir" "$(du -sh "$dir" 2>/dev/null | cut -f1)"
    else
        printf '   %-24s    not found: %s\n' "$name/" "$dir"
    fi
done
if [ ${#found[@]} -eq 0 ]; then
    echo "== nothing found for run '$RUN_NAME' — check the name (runs: $(find run_all -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null)) ==" >&2
    exit 1
fi

EXCLUDES=()
[ "$INCLUDE_CHECKPOINTS" -eq 0 ] && EXCLUDES=(--exclude="checkpoints" --exclude="checkpoints_tuned")

OUT="$OUT_DIR/${RUN_NAME}_results.tar"
TMP_OUT="$OUT.partial"
echo "== writing $OUT =="
if ! tar --create --dereference --file="$TMP_OUT" "${EXCLUDES[@]}" -C "$STAGE" "${found[@]}"; then
    rm -f "$TMP_OUT"
    echo "== tar failed — nothing written ==" >&2
    exit 1
fi
mv "$TMP_OUT" "$OUT"
echo "== done: $OUT ($(du -h "$OUT" | cut -f1)) =="
echo "== it holds patient-level MIMIC-IV data: keep it where the Data Use Agreement allows =="
