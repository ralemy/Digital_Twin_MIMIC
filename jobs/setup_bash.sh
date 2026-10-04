#!/bin/bash
# =============================================================================
# Every location that differs between Alliance clusters (Nibi, Rorqual, ...)
# or between users is set HERE and nowhere else. Moving to another cluster,
# or handing the project to another user, means editing only this file —
# jobs/, config/, src/ and the docs use the variables below.
#
#   source jobs/setup_bash.sh   # set the variables in this shell (do this, or
#                               # add it to ~/.bashrc, before sbatch/salloc);
#                               # every job also does it (jobs/load_profile.sh)
#   bash jobs/setup_bash.sh     # once per cluster, on a login node: also create
#                               # the repo's links and build .venv
#
# The block is picked by $CC_CLUSTER, which the Alliance environment sets
# (nibi, rorqual, ...); without it (a workstation) the "local" block applies.
#
# The four locations:
#   DT_REPO           where the git repo is cloned — found from this file's
#                     own location, so it is right on any cluster, for any
#                     user, without editing
#   DT_MIMIC_DIR      where the MIMIC-IV files (hosp/, icu/) are downloaded
#                     <- symlink $DT_REPO/mimic-iv
#   DT_RESULTS_DIR    where results are recorded: the work directories of the
#                     two scopes, $DT_RESULTS_DIR/mimic-iv-twin-work (lean)
#                     and $DT_RESULTS_DIR/mimic-iv-twin-work-full (19 variables)
#   DT_OLLAMA_MODELS  where the Ollama models are downloaded (exported as
#                     OLLAMA_MODELS, which ollama reads)
#                     <- symlink $DT_REPO/ollama-models
# Also per cluster/user:
#   DT_ACCOUNT        Slurm account, exported as SBATCH_ACCOUNT/SALLOC_ACCOUNT
#                     (the job scripts carry no --account of their own)
#   DT_MODULES        modules the jobs load before activating .venv
#   DT_SYS_PYTHON     a python3 with PyYAML that works before any module is
#                     loaded (reads the profile and configs on login nodes)
#   DT_OLLAMA_BIN     directory holding the ollama binary (put first on PATH;
#                     see installing_ollama.md)
#   DT_CLUSTER        $CC_CLUSTER, or "local"
# =============================================================================

DT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DT_CLUSTER="${CC_CLUSTER:-local}"

case "$DT_CLUSTER" in
    nibi)
        DT_ACCOUNT=def-roudsari
        # StdEnv/2023 is the default; arrow is optional (requirements.txt):
        # add "gcc arrow/25.0.0" to give pandas pyarrow. Not between runs of
        # one experiment — string columns then read as `str`, not `object`.
        DT_MODULES="StdEnv/2023 python/3.11"
        DT_SYS_PYTHON=/usr/bin/python3
        DT_MIMIC_DIR="$HOME/physionet.org/files/mimiciv/3.1"
        DT_RESULTS_DIR="$DT_REPO"
        DT_OLLAMA_MODELS="${SCRATCH:-$HOME/scratch}/ollama-local/models"
        DT_OLLAMA_BIN="$HOME/ollama-local/bin"
        ;;
    rorqual)
        # Compute nodes have no internet: copy MIMIC-IV and the Ollama models
        # here from Nibi with Globus instead of running prep1/prep2. Check
        # these paths against where the transfer put them.
        DT_ACCOUNT=def-roudsari
        DT_MODULES="StdEnv/2023 python/3.11"
        DT_SYS_PYTHON=/usr/bin/python3
        DT_MIMIC_DIR="$HOME/projects/$DT_ACCOUNT/mimic-iv/3.1"
        DT_RESULTS_DIR="$DT_REPO"
        DT_OLLAMA_MODELS="${SCRATCH:-$HOME/scratch}/ollama-local/models"
        DT_OLLAMA_BIN="$HOME/ollama-local/bin"
        ;;
    local)
        DT_ACCOUNT=""                       # no Slurm
        DT_MODULES=""                       # no module system: plain python3 -m venv
        DT_SYS_PYTHON=python3
        DT_MIMIC_DIR="$HOME/mimic-iv"
        DT_RESULTS_DIR="$HOME"
        DT_OLLAMA_MODELS="$HOME/.ollama/models"
        DT_OLLAMA_BIN="$HOME/ollama-local/bin"
        ;;
    *)
        echo "== jobs/setup_bash.sh has no block for cluster '$DT_CLUSTER' — add one ==" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# Repo symlink -> variable holding its target. Always symlinks: the data
# itself never lives under these names in the repo.
DT_LINKS=(
    "mimic-iv:DT_MIMIC_DIR"
    "ollama-models:DT_OLLAMA_MODELS"
)

OLLAMA_MODELS="$DT_OLLAMA_MODELS"
AGENTIC_DT_PRJ="$DT_REPO"                   # older name of DT_REPO
case ":$PATH:" in
    *":$DT_OLLAMA_BIN:"*) ;;
    *) PATH="$DT_OLLAMA_BIN:$PATH" ;;
esac
export DT_REPO DT_CLUSTER DT_ACCOUNT DT_MODULES DT_SYS_PYTHON DT_MIMIC_DIR DT_RESULTS_DIR \
       DT_OLLAMA_MODELS DT_OLLAMA_BIN OLLAMA_MODELS AGENTIC_DT_PRJ PATH
if [ -n "$DT_ACCOUNT" ]; then
    SBATCH_ACCOUNT="$DT_ACCOUNT"
    SALLOC_ACCOUNT="$DT_ACCOUNT"
    export SBATCH_ACCOUNT SALLOC_ACCOUNT
fi

# --- one-time setup: only when run, not when sourced -------------------------
dt_make_links() {
    local entry name var target link
    for entry in "${DT_LINKS[@]}"; do
        name=${entry%%:*}; var=${entry#*:}
        target=${!var}; link="$DT_REPO/$name"
        if [ "$target" = "$link" ]; then
            echo "== $var must point outside the repo — $link is the link to it ==" >&2
            return 1
        fi
        if [ -e "$link" ] && [ ! -L "$link" ]; then
            echo "== $link is a real directory, not a link — move its contents to $target (\$$var), remove it, then run this again ==" >&2
            return 1
        fi
        mkdir -p "$target"
        ln -sfn "$target" "$link"
        echo "   $name -> $target"
    done
    mkdir -p "$DT_RESULTS_DIR"
    echo "   results: $DT_RESULTS_DIR/mimic-iv-twin-work{,-full}"
}

dt_make_venv() {
    local venv="$DT_REPO/.venv"
    if [ -x "$venv/bin/python" ]; then
        echo "   .venv exists ($("$venv/bin/python" --version 2>&1)) — left as is; to rebuild: rm -rf .venv && bash jobs/setup_bash.sh"
        return 0
    fi
    if [ -n "$DT_MODULES" ]; then
        command -v module > /dev/null || { echo "== 'module' isn't available in this shell — run from a login shell ==" >&2; return 1; }
        # shellcheck disable=SC2086  # DT_MODULES is a word list
        module load $DT_MODULES || return 1
        virtualenv --no-download "$venv" || return 1
        "$venv/bin/pip" install --no-index --upgrade pip || return 1
        "$venv/bin/pip" install --no-index -r "$DT_REPO/requirements.txt" || return 1
    else
        python3 -m venv "$venv" || return 1
        "$venv/bin/pip" install --upgrade pip || return 1
        "$venv/bin/pip" install -r "$DT_REPO/requirements.txt" || return 1
    fi
    echo "   .venv built ($("$venv/bin/python" --version 2>&1))"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -euo pipefail
    echo "== cluster=$DT_CLUSTER  account=${DT_ACCOUNT:-none}  repo=$DT_REPO =="
    echo "== links =="
    dt_make_links
    echo "== virtual environment =="
    dt_make_venv
    echo "== done. Before sbatch/salloc in a new shell: source jobs/setup_bash.sh =="
fi
