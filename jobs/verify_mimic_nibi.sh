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
# Settings (project directory, data_root) come from your profile,
# ~/.config/dt_profile.yml (see config/profile.sample.yml and README,
# 'Your profile'). To use another profile, add --profile <file> anywhere
# in the job's arguments. Submit from the project directory:
#   cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/verify_mimic_nibi.sh [path/to/mimic-iv]
# The MIMIC-IV directory defaults to <data_root>/mimic-iv if omitted.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-verify-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Settings come from the profile (~/.config/dt_profile.yml, or --profile
# <file> among this job's arguments) — see jobs/load_profile.sh. The
# remaining arguments are this job's own.
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the project directory: cd <project> && sbatch jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

MIMIC_DIR="${1:-$PROJECT/mimic-iv}"
cd "$MIMIC_DIR"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  mimic_dir=$MIMIC_DIR =="

# --ignore-missing: only check the files actually downloaded (hosp/ and icu/
# here; SHA256SUMS.txt also lists files this project doesn't use).
status=0
sha256sum --check --ignore-missing SHA256SUMS.txt || status=$?

echo "== job $SLURM_JOB_ID finished at $(date) (exit $status) =="
exit $status
