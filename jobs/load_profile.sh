#!/bin/bash
# =============================================================================
# Settings loader shared by every job in jobs/ — SOURCED, not submitted.
#
# Each job does, right after `set -euo pipefail`:
#   source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh"
#   load_profile "$@" || exit 1
#   set -- "${JOB_ARGS[@]}"
# so submit from the repository base (`cd "$DT_REPO" && sbatch jobs/...`).
#
# load_profile:
#   - sources jobs/setup_bash.sh: the cluster's paths (DT_REPO, DT_MIMIC_DIR,
#     DT_OLLAMA_MODELS, DT_MODULES, ... — see that file), OLLAMA_MODELS, PATH;
#   - takes the job's arguments, removes `--profile <file>` (or
#     `--profile=<file>`) from them — default ~/.config/dt_profile.yml — and
#     leaves the rest in JOB_ARGS for the job;
#   - reads the PhysioNet credentials from the profile (see
#     config/profile.sample.yml), which only prep1 needs:
#       PHYSIONET_USERNAME, PHYSIONET_PASSWORD  (NOT exported — they stay in
#                                                the job's shell, never in
#                                                child processes)
#       PROFILE_FILE                            the profile that was read
#     The default profile may be missing (credentials stay empty); one named
#     with --profile must exist. A `paths:` section in it is ignored: paths
#     come from jobs/setup_bash.sh only.
#
# Uses $DT_SYS_PYTHON (has PyYAML) so it works before any module/venv is
# loaded. Returns non-zero with a message on any problem, so it's also safe
# to try in an interactive shell:
#   source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
# =============================================================================

load_profile() {
    local profile="$HOME/.config/dt_profile.yml" explicit=0 here parsed perms
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # shellcheck source=jobs/setup_bash.sh
    source "$here/jobs/setup_bash.sh" || return 1

    JOB_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --profile)
                [ $# -ge 2 ] || { echo "== --profile needs a file path ==" >&2; return 1; }
                profile="$2"; explicit=1; shift 2 ;;
            --profile=*)
                profile="${1#--profile=}"; explicit=1; shift ;;
            *)
                JOB_ARGS+=("$1"); shift ;;
        esac
    done
    profile="${profile/#\~/$HOME}"

    PHYSIONET_USERNAME=""
    PHYSIONET_PASSWORD=""
    PROFILE_FILE=""
    if [ -f "$profile" ]; then
        parsed=$("$DT_SYS_PYTHON" - "$profile" <<'PY'
import shlex, sys
import yaml

path = sys.argv[1]
try:
    with open(path) as f:
        prof = yaml.safe_load(f) or {}
except (OSError, yaml.YAMLError) as e:
    sys.exit(f"== could not read profile {path}: {e} ==")

if prof.get("paths"):
    print(f"== note: the paths: section of {path} is ignored — paths come from jobs/setup_bash.sh ==",
          file=sys.stderr)
physionet = prof.get("physionet") or {}
for key, value in {"PHYSIONET_USERNAME": physionet.get("username"),
                   "PHYSIONET_PASSWORD": physionet.get("password")}.items():
    print(f"{key}={shlex.quote(str(value or ''))}")
PY
        ) || return 1
        eval "$parsed"
        PROFILE_FILE="$profile"

        # A password in a file others can read isn't private: refuse it.
        if [ -n "$PHYSIONET_PASSWORD" ]; then
            perms=$(stat -L -c %a "$profile")
            if (( 8#$perms & 077 )); then
                echo "== profile '$profile' holds a password but is readable by others (mode $perms) — run: chmod 600 $profile ==" >&2
                unset PHYSIONET_PASSWORD
                return 1
            fi
        fi
    elif [ "$explicit" -eq 1 ]; then
        echo "== profile '$profile' not found ==" >&2
        return 1
    fi
    export -n PHYSIONET_USERNAME PHYSIONET_PASSWORD 2>/dev/null || true

    echo "== cluster=$DT_CLUSTER  repo=$DT_REPO  profile=${PROFILE_FILE:-none} =="
    echo "==   mimic=$DT_MIMIC_DIR  results=$DT_RESULTS_DIR  ollama_models=$OLLAMA_MODELS  modules=${DT_MODULES:-none} =="
}
