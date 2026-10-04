#!/bin/bash
# =============================================================================
# Slurm batch job — verify the downloaded MIMIC-IV files against PhysioNet's
# SHA256SUMS.txt. Run this if step 2 fails with DuckDB's
# "IO Error: Input is not a GZIP stream", which means one of the .csv.gz
# files is truncated or corrupted (typically an interrupted/resumed download).
# Single-core and I/O bound; checksumming ~10 GB of files takes a few minutes.
#
# Any line in the output reporting FAILED is a file to re-download from
# PhysioNet (delete it first rather than resuming the partial file).
#
#
# Locations, account and modules come from your profile
# (~/.config/dt_profile.yml), read by jobs/setup_bash.sh. Submit from the repository base:
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   sbatch jobs/verify_mimic_nibi.sh [path/to/mimic-iv]
# The MIMIC-IV directory defaults to $DT_MIMIC_DIR (paths.mimic_dir in your profile) if omitted.
# =============================================================================
#SBATCH --job-name=mimic-twin-verify-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -euo pipefail

# Settings come from your profile, through jobs/load_profile.sh; the
# job's arguments (minus any --profile <file>) are its own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

MIMIC_DIR="${1:-$DT_MIMIC_DIR}"
cd "$MIMIC_DIR"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=${SLURM_JOB_ACCOUNT:-$DT_ACCOUNT}  user=$(whoami)  mimic_dir=$MIMIC_DIR =="

# --ignore-missing: only check the files actually downloaded (hosp/ and icu/
# here; SHA256SUMS.txt also lists files this project doesn't use).
status=0
sha256sum --check --ignore-missing SHA256SUMS.txt || status=$?

echo "== job $SLURM_JOB_ID finished at $(date) (exit $status) =="
exit $status
