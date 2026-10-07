#!/bin/bash
# =============================================================================
# Pack one jobs/run_all.sh run into <run_name>_results.tar — run on a LOGIN
# NODE (it only reads and archives files):
#   bash jobs/pack_run.sh <run_name> [-o <output dir>] [--include-checkpoints] [--profile <file>]
#
# The archive holds, for whatever exists of the run:
#   logs/                       $DT_LOGS_ROOT/<run>/ (driver log, job .out
#                               files, Ollama logs)
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

usage() { sed -n '3,23p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

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

# <name in the archive>:<directory of this run>
PARTS=(
    "logs:$DT_LOGS_ROOT/$RUN_NAME"
    "mimic-iv-twin-work:$DT_RESULTS_DIR/mimic-iv-twin-work/$RUN_NAME"
    "mimic-iv-twin-work-full:$DT_RESULTS_DIR/mimic-iv-twin-work-full/$RUN_NAME"
    "tuned_configs:$DT_REPO/tuned_configs/$RUN_NAME"
)

# A staging directory of links named as in the archive; tar --dereference
# stores what they point to under those names.
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pack_run.XXXXXX") || exit 1
cleanup() { rm -f "$STAGE"/*; rmdir "$STAGE"; }
trap cleanup EXIT

found=()
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
