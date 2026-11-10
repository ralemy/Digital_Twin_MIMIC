#!/bin/bash
# =============================================================================
# Slurm batch job — Ollama throughput benchmark: OLLAMA_NUM_PARALLEL=8 and
# OLLAMA_FLASH_ATTENTION=1 vs the current 4 / off, before using them for the
# long runs (docs/runtime_estimates.md, section 3).
#
#   cd "$DT_REPO"     # after the setup (README, section 1): DT_REPO, SBATCH_ACCOUNT come from ~/.bashrc
#   bash jobs/submit.sh jobs/bench_ollama.sh                       # lean config
#   bash jobs/submit.sh jobs/bench_ollama.sh config/config_alliance_lean.yaml
#
# For each setting it restarts Ollama with it and forecasts the same 64
# validation patients (the tuning subset; the test split is never read) with
# full_pipeline on the default model (llm.model; qwen2.5:32b when this was
# run, Med42-70B since tri_lean_exp2):
#   np4_fa0  OLLAMA_NUM_PARALLEL=4, flash attention off   (current setting)
#   np4_fa1  4, on
#   np8_fa0  8, off
#   np8_fa1  8, on                                       (the candidate)
# Under np8_fa1 it also checks the models still fit in GPU memory at 8
# request slots: Med42-70B (the largest; llm.model) and qwen2.5 at the lean num_ctx
# (8192) and at the 19-variable scope's num_ctx (12288).
#
# Output: one line per run in <results_dir>/ollama_bench/bench-<job id>.jsonl,
# a summary table at the end of this job's .out (or later:
# python src/bench_ollama.py --config-file <config> --summary), and one
# Ollama log per setting, $DT_LOG_DIR/ollama-<job id>-<setting>.log.
#
# Reading it: "min/450" is the projected time of one LLM condition on the
# 450 test patients (measured: ~84 min for full_pipeline at np4_fa0);
# fallbacks/filled/sMAPE should be similar across settings (at temperature
# 0.2 and 64 patients, a few % of sMAPE is noise). To adopt a setting, set
# performance.llm_max_concurrent_requests (= OLLAMA_NUM_PARALLEL) and
# performance.ollama_flash_attention in the config — every job's Ollama
# server reads them (jobs/ollama_lib.sh); neither changes checkpoint
# fingerprints — and re-estimate with docs/runtime_estimates.md.
#
# Time: ~45-50 min estimated (4 x (~1.5 min startup + 7-11 min forecasting)
# + probes); the limit is 1 h more than that.
# =============================================================================
#SBATCH --job-name=mimic-twin-bench-ollama
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out   # relative to the repo base; run_all.sh overrides it

set -uo pipefail     # no -e: one failing setting shouldn't stop the others

source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base: cd \$DT_REPO && bash jobs/submit.sh jobs/<job>.sh ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"

cd "$DT_REPO" || exit 1
CONFIG="${1:-config/config_alliance_lean.yaml}"
echo "== job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date) — config=$CONFIG =="

module load $DT_MODULES          # environment.modules in your profile
source "$DT_REPO/.venv/bin/activate"
source jobs/ollama_lib.sh

bench() { python src/bench_ollama.py --config-file "$CONFIG" "$@" || echo "== FAILED: bench_ollama.py $* ==" >&2; }

# run_setting <label> <num_parallel> <flash_attention 0|1> [probes...]
run_setting() {
    local label=$1 np=$2 fa=$3
    echo "== setting $label: OLLAMA_NUM_PARALLEL=$np OLLAMA_FLASH_ATTENTION=$fa at $(date) =="
    if ! start_ollama "$CONFIG" "$label" "$np" "$fa"; then
        echo "== FAILED: ollama didn't start for $label ==" >&2
        stop_ollama
        return
    fi
    if [ "$label" = np4_fa0 ]; then
        require_models "$CONFIG" all || { stop_ollama; exit 1; }
    fi
    bench --label "$label" --parallel "$np"
    if [ "$label" = np8_fa1 ]; then
        bench --label "$label" --probe default --num-ctx 8192
        bench --label "$label" --probe default --num-ctx 12288
        bench --label "$label" --probe qwen2_5_32b --num-ctx 12288
    fi
    stop_ollama
}

run_setting np4_fa0 4 0
run_setting np4_fa1 4 1
run_setting np8_fa0 8 0
run_setting np8_fa1 8 1

echo "== summary =="
python src/bench_ollama.py --config-file "$CONFIG" --summary
echo "== job ${SLURM_JOB_ID:-local} finished at $(date) =="
