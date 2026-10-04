#!/bin/bash
# =============================================================================
# Slurm batch job — download MIMIC-IV (hosp/ and icu/ modules) from PhysioNet
# into $DT_MIMIC_DIR (jobs/setup_bash.sh), so the result is
#   $DT_MIMIC_DIR/hosp/*.csv.gz
#   $DT_MIMIC_DIR/icu/*.csv.gz
# which the repo's mimic-iv link (the configs' mimic_root) points to.
#
# Uses wget, as PhysioNet documents (PhysioNet answers curl with 403
# Forbidden, see mimic-twin-download-22971979.out). Instead of crawling the
# directory listings recursively (`wget -r`, which in job 22972423 fetched
# the top-level listing and then followed none of its links), it takes the
# file list from PhysioNet's SHA256SUMS.txt and asks wget for exactly the
# hosp/ and icu/ files that are missing locally.
#
# Every file is checked against SHA256SUMS.txt before and after the
# download. Files that fail the checksum (e.g. a corrupted labevents.csv.gz)
# are deleted, so they count as missing and are fetched again from scratch.
# Files that pass are left alone. Re-submitting this job after a timeout or
# failure simply continues where it left off.
#
# All Nibi nodes have internet access (Alliance docs, Nibi > Site specifics),
# so this runs fine as a regular job. Single-core and network bound.
#
# The destination comes from jobs/setup_bash.sh; the PhysioNet username and
# password from your profile, ~/.config/dt_profile.yml (see
# config/profile.sample.yml and README, 'Your profile'). To use another
# profile, add --profile <file> anywhere in the job's arguments.
# The credentials stay in your private profile (chmod 600 — the job refuses
# it otherwise) and are never exported to child processes.
# They are written only to a mode-600 wgetrc file in $SLURM_TMPDIR (removed
# at exit) that wget reads via $WGETRC, so the password never appears on a
# command line / in `ps` output.
#
# Submit from the repository base:
#   cd "$DT_REPO"     # with jobs/setup_bash.sh sourced (sets DT_REPO, SBATCH_ACCOUNT)
#   sbatch jobs/prep1_download_mimic_nibi.sh [mimic-iv version, default 3.1]
# =============================================================================
#SBATCH --job-name=mimic-twin-download
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Paths come from jobs/setup_bash.sh, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

: "${PHYSIONET_USERNAME:?physionet.username is empty in ${PROFILE_FILE:-~/.config/dt_profile.yml (not found)} — fill it in (see config/profile.sample.yml)}"
: "${PHYSIONET_PASSWORD:?physionet.password is empty in ${PROFILE_FILE:-~/.config/dt_profile.yml (not found)} — fill it in (see config/profile.sample.yml)}"

VERSION="${1:-3.1}"
BASE_URL="https://physionet.org/files/mimiciv/$VERSION/"
DEST="$DT_MIMIC_DIR"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  dest=$DEST  mimic-iv=$VERSION =="

mkdir -p "$DEST"
cd "$DEST"

# Credentials -> private wgetrc. wgetrc takes the rest of the line as the
# value, so no quoting is needed (leading/trailing spaces are trimmed).
WGETRC_FILE="${SLURM_TMPDIR:-$(mktemp -d)}/physionet.wgetrc"
trap 'rm -f "$WGETRC_FILE"' EXIT
( umask 077
  printf 'http_user = %s\nhttp_password = %s\n' "$PHYSIONET_USERNAME" "$PHYSIONET_PASSWORD" > "$WGETRC_FILE" )
unset PHYSIONET_PASSWORD
export WGETRC="$WGETRC_FILE"

# Check hosp/ and icu/ files that are present against SHA256SUMS.txt and
# delete any that fail. Prints the number of failures.
remove_bad_files() {
    [[ -f SHA256SUMS.txt ]] || { echo 0; return; }
    local n=0 path
    while IFS= read -r path; do
        echo "checksum FAILED, deleting: $path" >&2
        rm -f "$path"
        n=$((n + 1))
    done < <(grep -E ' \*?(hosp|icu)/' SHA256SUMS.txt \
             | sha256sum --check --ignore-missing 2>/dev/null \
             | sed -n 's/: FAILED$//p')
    echo "$n"
}

# Print the hosp/ and icu/ paths listed in SHA256SUMS.txt that don't exist
# locally, one per line.
list_missing() {
    local path
    while read -r _ path; do
        path="${path#\*}"   # sha256sum's binary-mode marker, if present
        [[ -f "$path" ]] || echo "$path"
    done < <(grep -E ' \*?(hosp|icu)/' SHA256SUMS.txt)
}

# -4 : IPv4 only (the cluster's default wget alias does the same)
# -nv: one line per file instead of a progress bar in the log
WGET=(wget -4 -nv --tries=5 --waitretry=30)

# Refresh the checksum list and top-level files. -N only re-downloads them
# if PhysioNet's copy is newer. If this fails but a local SHA256SUMS.txt
# exists, carry on with the local one.
echo "== fetching SHA256SUMS.txt, LICENSE.txt, CHANGELOG.txt =="
for f in SHA256SUMS.txt LICENSE.txt CHANGELOG.txt; do
    "${WGET[@]}" -N "$BASE_URL$f" || echo "WARNING: could not fetch $f (wget exit $?)" >&2
done
[[ -f SHA256SUMS.txt ]] || { echo "SHA256SUMS.txt could not be downloaded — check credentials / MIMIC-IV access" >&2; exit 1; }
expected=$(grep -cE ' \*?(hosp|icu)/' SHA256SUMS.txt)
mkdir -p hosp icu

echo "== checking existing files =="
echo "== $(remove_bad_files) existing file(s) removed for re-download =="

mapfile -t TO_FETCH < <(list_missing)
echo "== ${#TO_FETCH[@]} of $expected hosp/icu files to download =="
for path in "${TO_FETCH[@]}"; do
    echo "downloading  $path"
    # Download next to the target and rename only when complete, so an
    # interrupted transfer never leaves a partial file under the real name.
    rm -f "$path.part"
    if "${WGET[@]}" -O "$path.part" "$BASE_URL$path"; then
        mv -f "$path.part" "$path"
    else
        echo "download failed for $path (wget exit $?)" >&2
        rm -f "$path.part"
    fi
done

echo "== verifying checksums =="
bad=$(remove_bad_files)
mapfile -t STILL_MISSING < <(list_missing)
missing=${#STILL_MISSING[@]}
for path in "${STILL_MISSING[@]}"; do echo "missing: $path" >&2; done

echo "== disk usage: $(du -sh "$DEST" | cut -f1) in $DEST =="
if (( bad + missing )); then
    echo "== $missing file(s) missing after download ($bad of them failed the checksum and were deleted) — re-submit this job to retry ==" >&2
    exit 1
fi
echo "== all $expected hosp/icu files verified; job $SLURM_JOB_ID finished at $(date) =="
