#!/bin/bash
# =============================================================================
# Sets every location and setting that differs between clusters or users,
# from YOUR PROFILE — ~/.config/dt_profile.yml (template:
# config/profile.sample.yml). The profile is the only file you edit to set
# the project up; nothing in jobs/, config/, src/ or this file needs changing.
#
# Setting up a fresh clone (once per cluster, on a login node):
#   1. cp config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 it,
#      then fill it in
#   2. bash jobs/setup_bash.sh
#   3. source ~/.bashrc
# Step 2 creates the repo's symlinks, installs Ollama (no root needed),
# builds .venv and writes the final values of the variables below into a
# marked block of ~/.bashrc. It logs everything it prints to setup.lock in
# the repo base and refuses to run while that file exists; to run it again
# (e.g. after changing the profile): rm setup.lock && bash jobs/setup_bash.sh
# A failed run leaves no lock: its log is kept as setup-failed-<time>.log.
#
#   source jobs/setup_bash.sh   # only reads the profile and sets the variables
#                               # in this shell — every job does this, through
#                               # jobs/load_profile.sh; never blocked by the lock
#
# Alliance clusters don't share home directories, so each cluster has its
# own ~/.config/dt_profile.yml. To read another file, set DT_PROFILE=<file>
# (jobs: --profile <file>).
#
# Variables set (profile key in brackets):
#   DT_REPO           where the git repo is cloned — found from this file's
#                     own location, never configured
#   DT_MIMIC_DIR      [paths.mimic_dir] where the MIMIC-IV files (hosp/, icu/)
#                     are downloaded               <- symlink $DT_REPO/mimic-iv
#   DT_RESULTS_DIR    [paths.results_dir] where results are recorded:
#                     $DT_RESULTS_DIR/mimic-iv-twin-work (lean) and
#                     $DT_RESULTS_DIR/mimic-iv-twin-work-full (full)
#   DT_OLLAMA_MODELS  [paths.ollama_models] where the Ollama models are
#                     downloaded; exported as OLLAMA_MODELS too
#                                                  <- symlink $DT_REPO/ollama-models
#   DT_OLLAMA_BIN     [paths.ollama_bin, default ~/ollama-local/bin] the
#                     directory holding the ollama binary (put first on PATH);
#                     Ollama is installed into its parent directory
#   DT_OLLAMA_VERSION [environment.ollama_version, default 0.34.4] the Ollama
#                     release installed
#   DT_ACCOUNT        [slurm.account] Slurm account, exported as
#                     SBATCH_ACCOUNT/SALLOC_ACCOUNT (the job scripts carry no
#                     --account of their own)
#   DT_MODULES        [environment.modules, default "StdEnv/2023 python/3.11"
#                     on a cluster] modules the jobs load before .venv
#   DT_SYS_PYTHON     a python3 with PyYAML that works before any module is
#                     loaded (reads the profile); found automatically
#   DT_CLUSTER        $CC_CLUSTER (nibi, rorqual, ...), or "local"
#   DT_PROFILE        the profile that was read
#   DT_LOG_DIR        where jobs write their Ollama logs: $DT_REPO/logs unless
#                     already set (jobs/run_all.sh sets a directory per run);
#                     Slurm's own .out files go to logs/ via #SBATCH --output
# The PhysioNet credentials in the profile are NOT read here — only by the
# download job, through jobs/load_profile.sh.
# =============================================================================

DT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DT_CLUSTER="${CC_CLUSTER:-local}"
DT_PROFILE="${DT_PROFILE:-$HOME/.config/dt_profile.yml}"
DT_PROFILE="${DT_PROFILE/#\~/$HOME}"
if [ -z "${DT_SYS_PYTHON:-}" ]; then
    if /usr/bin/python3 -c 'import yaml' 2> /dev/null; then
        DT_SYS_PYTHON=/usr/bin/python3
    else
        DT_SYS_PYTHON=python3
    fi
fi

# --- run as `bash jobs/setup_bash.sh`: lock and log first -------------------
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -euo pipefail
    DT_LOCK="$DT_REPO/setup.lock"
    # noclobber makes creating the lock atomic: two runs can't both get it.
    if ! ( set -o noclobber; : > "$DT_LOCK" ) 2> /dev/null; then
        echo "== setup has already run in this clone ($DT_LOCK holds its log) — refusing to run again." >&2
        echo "== To redo it, e.g. after changing your profile: rm $DT_LOCK && bash jobs/setup_bash.sh ==" >&2
        exit 1
    fi
    dt_setup_exit() {
        local rc=$? failed
        if [ "$rc" -ne 0 ]; then
            failed="$DT_REPO/setup-failed-$(date +%Y%m%d-%H%M%S).log"
            echo "== setup FAILED (exit $rc). Nothing is locked; this log is kept as $failed."
            echo "== Fix the cause above, then run bash jobs/setup_bash.sh again. =="
            mv -f "$DT_LOCK" "$failed"
        else
            echo "== setup finished at $(date '+%F %T'). Now run: source ~/.bashrc =="
        fi
    }
    trap dt_setup_exit EXIT
    exec > >(tee -a "$DT_LOCK") 2>&1
    echo "== setup started at $(date '+%F %T') by $USER on $(hostname) — repo $DT_REPO, profile $DT_PROFILE =="
fi

# Reads the profile and sets the DT_* variables above; returns 1 with a
# message if the profile is missing, readable by others, or incomplete.
dt_read_profile() {
    local perms parsed
    if [ ! -f "$DT_PROFILE" ]; then
        echo "== no profile at $DT_PROFILE — create it from the template, then fill it in (README, section 1):" >&2
        echo "==   cp $DT_REPO/config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 ~/.config/dt_profile.yml ==" >&2
        return 1
    fi
    perms=$(stat -L -c %a "$DT_PROFILE")
    if (( 8#$perms & 077 )); then
        echo "== profile $DT_PROFILE is readable by others (mode $perms) — it holds your PhysioNet password: chmod 600 $DT_PROFILE ==" >&2
        return 1
    fi
    parsed=$(DT_REPO="$DT_REPO" "$DT_SYS_PYTHON" - "$DT_PROFILE" "$DT_CLUSTER" <<'PY'
import os, shlex, sys
import yaml

path, cluster = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        prof = yaml.safe_load(f) or {}
except (OSError, yaml.YAMLError) as e:
    sys.exit(f"== could not read profile {path}: {e} ==")

problems = []

def get(section, key):
    value = (prof.get(section) or {}).get(key)
    return "" if value is None else str(value).strip()

def location(key, default=""):
    value = get("paths", key) or default
    if not value:
        problems.append(f"paths.{key} is empty")
        return ""
    # ~, $HOME, $SCRATCH, $USER, $DT_REPO, ... are allowed.
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        problems.append(f"paths.{key} uses an unset variable: {value}")
    elif not os.path.isabs(expanded):
        problems.append(f"paths.{key} must be an absolute path (it may start with ~ or $HOME): {value}")
    return os.path.normpath(expanded)

on_cluster = cluster != "local"
values = {
    "DT_ACCOUNT": get("slurm", "account"),
    "DT_MIMIC_DIR": location("mimic_dir"),
    "DT_RESULTS_DIR": location("results_dir"),
    "DT_OLLAMA_MODELS": location("ollama_models"),
    "DT_OLLAMA_BIN": location("ollama_bin", "~/ollama-local/bin"),
    "DT_OLLAMA_VERSION": get("environment", "ollama_version") or "0.34.4",
    "DT_MODULES": get("environment", "modules") or ("StdEnv/2023 python/3.11" if on_cluster else ""),
}
if on_cluster and not values["DT_ACCOUNT"]:
    problems.append("slurm.account is empty")
if os.path.basename(values["DT_OLLAMA_BIN"]) != "bin":
    problems.append(f"paths.ollama_bin must end in /bin (the Ollama archive unpacks bin/ and lib/): {values['DT_OLLAMA_BIN']}")
if problems:
    print(f"== profile {path} is incomplete — fill it in (see config/profile.sample.yml):", file=sys.stderr)
    for p in problems:
        print(f"==   {p}", file=sys.stderr)
    paths = prof.get("paths") or {}
    if "data_root" in paths or "project_dir" in paths:
        print("==   (it uses the old format: project_dir/data_root are replaced by mimic_dir/results_dir)", file=sys.stderr)
    sys.exit(1)
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
    ) || return 1
    eval "$parsed"
}

if ! dt_read_profile; then
    return 1 2> /dev/null || exit 1
fi

# Repo symlink -> variable holding its target. Always symlinks: the data
# itself never lives under these names in the repo.
DT_LINKS=(
    "mimic-iv:DT_MIMIC_DIR"
    "ollama-models:DT_OLLAMA_MODELS"
)

OLLAMA_MODELS="$DT_OLLAMA_MODELS"
case ":$PATH:" in
    *":$DT_OLLAMA_BIN:"*) ;;
    *) PATH="$DT_OLLAMA_BIN:$PATH" ;;
esac
# Every variable written to ~/.bashrc by the setup, in this order.
DT_EXPORTS=(DT_REPO DT_CLUSTER DT_PROFILE DT_ACCOUNT DT_MODULES DT_SYS_PYTHON DT_MIMIC_DIR DT_RESULTS_DIR
            DT_OLLAMA_MODELS DT_OLLAMA_BIN DT_OLLAMA_VERSION OLLAMA_MODELS)
export "${DT_EXPORTS[@]}" PATH
# Not in ~/.bashrc: run_all.sh points it at its own directory for each run.
DT_LOG_DIR="${DT_LOG_DIR:-$DT_REPO/logs}"
export DT_LOG_DIR
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
    mkdir -p "$DT_RESULTS_DIR" "$DT_REPO/logs"
    echo "   results: $DT_RESULTS_DIR/mimic-iv-twin-work{,-full}"
    echo "   job logs: $DT_REPO/logs/"
}

# Installs the Ollama release DT_OLLAMA_VERSION into the parent of
# DT_OLLAMA_BIN (bin/ and lib/), unless an ollama binary is already there.
# No root needed; login nodes have internet access. ~1.4 GB download.
dt_install_ollama() {
    local root archive url
    if [ -x "$DT_OLLAMA_BIN/ollama" ]; then
        echo "   already installed: $DT_OLLAMA_BIN/ollama ($("$DT_OLLAMA_BIN/ollama" --version 2>&1 | grep -o 'version is .*' || echo 'version unknown')) — left as is"
        return 0
    fi
    root=$(dirname "$DT_OLLAMA_BIN")
    archive="$root/ollama-$DT_OLLAMA_VERSION.tar.zst"
    url="https://github.com/ollama/ollama/releases/download/v$DT_OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
    mkdir -p "$root"
    echo "   downloading Ollama $DT_OLLAMA_VERSION (~1.4 GB) from $url"
    curl -fsSL --retry 3 -o "$archive.part" "$url" || { rm -f "$archive.part"; echo "== Ollama download failed ==" >&2; return 1; }
    mv -f "$archive.part" "$archive"
    if ! tar --zstd -xf "$archive" -C "$root" 2> /dev/null; then
        zstd -dc "$archive" | tar -xf - -C "$root" || { echo "== could not unpack $archive ==" >&2; return 1; }
    fi
    rm -f "$archive"
    [ -x "$DT_OLLAMA_BIN/ollama" ] || { echo "== $DT_OLLAMA_BIN/ollama not found after unpacking into $root ==" >&2; return 1; }
    echo "   installed: $DT_OLLAMA_BIN/ollama ($(du -sh "$root" | cut -f1) in $root)"
}

dt_make_venv() {
    local venv="$DT_REPO/.venv"
    if [ -x "$venv/bin/python" ]; then
        echo "   .venv exists ($("$venv/bin/python" --version 2>&1)) — left as is; to rebuild: rm -rf .venv, then run the setup again"
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

# Writes the final values into a marked block at the end of ~/.bashrc,
# replacing the block a previous setup wrote (a backup is kept as
# ~/.bashrc.dt-backup). Lines elsewhere in ~/.bashrc that set the same
# variables are reported, not changed.
dt_write_bashrc() {
    local rc="$HOME/.bashrc" var tmp bin_q others
    local begin="# >>> mimic-iv digital twin — written by jobs/setup_bash.sh >>>"
    local end="# <<< mimic-iv digital twin <<<"
    touch "$rc"
    cp -p "$rc" "$rc.dt-backup"
    tmp=$(mktemp "$rc.XXXXXX")
    awk -v b="$begin" -v e="$end" '$0 == b {skip = 1; next} $0 == e {skip = 0; next} !skip' "$rc" > "$tmp"
    others=$(grep -nE '^[[:space:]]*export[[:space:]]+(DT_[A-Z_]+|OLLAMA_MODELS|SBATCH_ACCOUNT|SALLOC_ACCOUNT|AGENTIC_DT_PRJ)=|ollama-local/bin' "$tmp" || true)
    bin_q=$(printf %q "$DT_OLLAMA_BIN")
    {
        echo "$begin"
        echo "# $(date '+%F %T') from $DT_PROFILE — change the profile and re-run the setup, not this block"
        for var in "${DT_EXPORTS[@]}"; do
            echo "export $var=$(printf %q "${!var}")"
        done
        if [ -n "$DT_ACCOUNT" ]; then
            echo "export SBATCH_ACCOUNT=$(printf %q "$DT_ACCOUNT") SALLOC_ACCOUNT=$(printf %q "$DT_ACCOUNT")"
        fi
        echo "case \":\$PATH:\" in *:$bin_q:*) ;; *) export PATH=$bin_q:\"\$PATH\" ;; esac"
        echo "$end"
    } >> "$tmp"
    cat "$tmp" > "$rc"
    rm -f "$tmp"
    echo "   ~/.bashrc: block written (previous version kept as ~/.bashrc.dt-backup)"
    if [ -n "$others" ]; then
        echo "   NOTE: these lines elsewhere in ~/.bashrc set related variables; the block comes"
        echo "   after them and wins, but you may want to delete them:"
        echo "$others" | sed 's/^/     line /'
    fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "== cluster=$DT_CLUSTER  account=${DT_ACCOUNT:-none} =="
    echo "== 1/4 links =="
    dt_make_links
    echo "== 2/4 Ollama =="
    dt_install_ollama
    echo "== 3/4 virtual environment =="
    dt_make_venv
    echo "== 4/4 ~/.bashrc =="
    dt_write_bashrc
fi
