#!/bin/bash
# =============================================================================
# Profile loader shared by every job in jobs/ — SOURCED, not submitted.
#
# Each job does, right after `set -euo pipefail`:
#   source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh"
#   load_profile "$@" || exit 1
#   set -- "${JOB_ARGS[@]}"
# so submit from the project directory (`cd <project> && sbatch jobs/...`).
#
# load_profile takes the job's arguments, removes `--profile <file>` (or
# `--profile=<file>`) from them — default ~/.config/dt_profile.yml — and
# leaves the rest in JOB_ARGS for the job. It then reads the profile (see
# config/profile.sample.yml) and sets:
#   AGENTIC_DT_PRJ      paths.project_dir   (exported)
#   PROJECT             paths.data_root     (exported; the configs' $PROJECT)
#   OLLAMA_MODELS       paths.ollama_models (exported; read by ollama)
#   PATH                paths.ollama_bin prepended (exported)
#   PHYSIONET_USERNAME  physionet.username  (NOT exported)
#   PHYSIONET_PASSWORD  physionet.password  (NOT exported — stays in the job's
#                                            shell, never in child processes)
#   PROFILE_FILE        the profile that was read
# The profile always wins over anything already in the environment (e.g.
# old exports in ~/.bashrc), so a job's settings come from one place.
#
# Uses /usr/bin/python3 (has PyYAML) so it works before any module/venv is
# loaded. Returns non-zero with a message on any problem, so it's also safe
# to try in an interactive shell:
#   source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
# =============================================================================

load_profile() {
    local profile="$HOME/.config/dt_profile.yml" here parsed perms
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    JOB_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --profile)
                [ $# -ge 2 ] || { echo "== --profile needs a file path ==" >&2; return 1; }
                profile="$2"; shift 2 ;;
            --profile=*)
                profile="${1#--profile=}"; shift ;;
            *)
                JOB_ARGS+=("$1"); shift ;;
        esac
    done
    profile="${profile/#\~/$HOME}"

    if [ ! -f "$profile" ]; then
        echo "== profile '$profile' not found. Create it once (see README, 'Your profile'):" >&2
        echo "==   cp $here/config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 ~/.config/dt_profile.yml" >&2
        echo "== or pass --profile <file> to the job. ==" >&2
        return 1
    fi

    parsed=$(/usr/bin/python3 - "$profile" "$here" <<'PY'
import os, shlex, sys
import yaml

path, here = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        prof = yaml.safe_load(f) or {}
except (OSError, yaml.YAMLError) as e:
    sys.exit(f"== could not read profile {path}: {e} ==")

paths = prof.get("paths") or {}
physionet = prof.get("physionet") or {}
if not paths.get("ollama_models"):
    sys.exit(f"== profile {path} has no paths.ollama_models — see config/profile.sample.yml ==")

def expand(v):
    return os.path.expanduser(os.path.expandvars(str(v)))

project_dir = expand(paths.get("project_dir") or here)
values = {
    "AGENTIC_DT_PRJ": project_dir,
    "PROJECT": expand(paths.get("data_root") or project_dir),
    "OLLAMA_MODELS": expand(paths["ollama_models"]),
    "DT_OLLAMA_BIN": expand(paths.get("ollama_bin") or "~/ollama-local/bin"),
    "PHYSIONET_USERNAME": str(physionet.get("username") or ""),
    "PHYSIONET_PASSWORD": str(physionet.get("password") or ""),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
    ) || return 1
    eval "$parsed"

    # A password in a file others can read isn't private: refuse it.
    if [ -n "$PHYSIONET_PASSWORD" ]; then
        perms=$(stat -L -c %a "$profile")
        if (( 8#$perms & 077 )); then
            echo "== profile '$profile' holds a password but is readable by others (mode $perms) — run: chmod 600 $profile ==" >&2
            unset PHYSIONET_PASSWORD
            return 1
        fi
    fi

    if [ ! -d "$AGENTIC_DT_PRJ" ]; then
        echo "== paths.project_dir '$AGENTIC_DT_PRJ' in $profile does not exist ==" >&2
        return 1
    fi
    if ! [ "$AGENTIC_DT_PRJ" -ef "$here" ]; then
        echo "== WARNING: paths.project_dir ($AGENTIC_DT_PRJ) is not the repo this job was submitted from ($here) ==" >&2
    fi

    case ":$PATH:" in
        *":$DT_OLLAMA_BIN:"*) ;;
        *) PATH="$DT_OLLAMA_BIN:$PATH" ;;
    esac
    export AGENTIC_DT_PRJ PROJECT OLLAMA_MODELS PATH
    export -n PHYSIONET_USERNAME PHYSIONET_PASSWORD 2>/dev/null || true
    PROFILE_FILE="$profile"

    echo "== profile: $PROFILE_FILE =="
    echo "==   project_dir=$AGENTIC_DT_PRJ  data_root=$PROJECT  ollama_models=$OLLAMA_MODELS =="
}
