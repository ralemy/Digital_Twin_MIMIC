#!/bin/bash
# =============================================================================
# Settings loader shared by every job in jobs/ — SOURCED, not submitted.
#
# Each job does, right after `set -euo pipefail`:
#   source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh"
#   load_profile "$@" || exit 1
#   set -- "${JOB_ARGS[@]}"
# so submit from the repository base (`cd "$DT_REPO" && bash jobs/submit.sh jobs/...`).
#
# load_profile:
#   - takes the job's arguments, removes `--profile <file>` (or
#     `--profile=<file>`) and `--log-dir <dir>` (DT_LOG_DIR, where the job's
#     Ollama log goes) from them and leaves the rest in JOB_ARGS for the
#     job. The profile is that file, else $DT_PROFILE, else
#     ~/.config/dt_profile.yml (template: config/profile.sample.yml);
#   - sources jobs/setup_bash.sh, which reads the profile's locations and
#     settings (DT_REPO, DT_MIMIC_DIR, DT_RESULTS_DIR, DT_OLLAMA_MODELS,
#     DT_ACCOUNT, DT_MODULES, ... — see that file), OLLAMA_MODELS and PATH,
#     and refuses a missing, incomplete or world-readable profile;
#   - reads the PhysioNet credentials from the same profile, which only
#     prep1 needs:
#       PHYSIONET_USERNAME, PHYSIONET_PASSWORD  (NOT exported — they stay in
#                                                the job's shell, never in
#                                                child processes)
#       PROFILE_FILE                            the profile that was read
#
# Uses $DT_SYS_PYTHON (has PyYAML) so it works before any module/venv is
# loaded. Returns non-zero with a message on any problem, so it's also safe
# to try in an interactive shell:
#   source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
# =============================================================================

load_profile() {
    local here parsed
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    JOB_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --profile)
                [ $# -ge 2 ] || { echo "== --profile needs a file path ==" >&2; return 1; }
                DT_PROFILE="$2"; shift 2 ;;
            --profile=*)
                DT_PROFILE="${1#--profile=}"; shift ;;
            --log-dir)
                # Where this job's Ollama log goes (jobs/run_all.sh: the run's
                # folder). An argument, not the environment: Trillium starts
                # jobs with a clean one (--export=NONE), so a DT_LOG_DIR set
                # by the driver never reached tri_lean_exp2's jobs.
                [ $# -ge 2 ] || { echo "== --log-dir needs a directory ==" >&2; return 1; }
                DT_LOG_DIR="$2"; export DT_LOG_DIR; shift 2 ;;
            *)
                JOB_ARGS+=("$1"); shift ;;
        esac
    done

    # shellcheck source=jobs/setup_bash.sh
    source "$here/jobs/setup_bash.sh" || return 1

    parsed=$("$DT_SYS_PYTHON" - "$DT_PROFILE" <<'PY'
import shlex, sys
import yaml

with open(sys.argv[1]) as f:
    physionet = (yaml.safe_load(f) or {}).get("physionet") or {}
for key, value in {"PHYSIONET_USERNAME": physionet.get("username"),
                   "PHYSIONET_PASSWORD": physionet.get("password")}.items():
    print(f"{key}={shlex.quote(str(value or ''))}")
PY
    ) || return 1
    eval "$parsed"
    PROFILE_FILE="$DT_PROFILE"
    export -n PHYSIONET_USERNAME PHYSIONET_PASSWORD 2>/dev/null || true

    echo "== cluster=$DT_CLUSTER  account=${DT_ACCOUNT:-none}  repo=$DT_REPO  profile=$PROFILE_FILE =="
    echo "==   mimic=$DT_MIMIC_DIR  results=$DT_RESULTS_DIR  ollama_models=$OLLAMA_MODELS  modules=${DT_MODULES:-none} =="
}
