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
# Reads the project directory from $AGENTIC_DT_PRJ (must be exported in the
# submitting shell — sbatch passes the submission environment through by
# default, so `export AGENTIC_DT_PRJ=...` before `sbatch` is enough).
#
# Submit with:
#   export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
#   sbatch jobs/verify_mimic_nibi.sh [path/to/mimic-iv]
# The MIMIC-IV directory defaults to $AGENTIC_DT_PRJ/mimic-iv if omitted.
# =============================================================================
#SBATCH --account=def-roudsari
#SBATCH --job-name=mimic-twin-verify-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${AGENTIC_DT_PRJ:?AGENTIC_DT_PRJ is not set — export it to the project directory before sbatch, e.g. export AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1}"
MIMIC_DIR="${1:-$AGENTIC_DT_PRJ/mimic-iv}"
cd "$MIMIC_DIR"

echo "== job $SLURM_JOB_ID starting on $(hostname) at $(date) =="
echo "== account=def-roudsari  user=$(whoami)  mimic_dir=$MIMIC_DIR =="

# --ignore-missing: only check the files actually downloaded (hosp/ and icu/
# here; SHA256SUMS.txt also lists files this project doesn't use).
status=0
sha256sum --check --ignore-missing SHA256SUMS.txt || status=$?

echo "== job $SLURM_JOB_ID finished at $(date) (exit $status) =="
exit $status
