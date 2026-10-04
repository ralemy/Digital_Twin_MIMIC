# How jobs get their settings: `jobs/setup_bash.sh` and `jobs/load_profile.sh`

This page walks through the two files every job in `jobs/` relies on: first
what each is for, then how a job uses them, then `load_profile` step by step,
a worked example and the error messages you can run into. Line numbers refer
to [`jobs/load_profile.sh`](../jobs/load_profile.sh).

---

## What they're for

Every job needs a few values that differ between Alliance clusters (Nibi,
Rorqual, ...) and between users:

| Location | Variable (set in `jobs/setup_bash.sh`) | Repo symlink |
|---|---|---|
| where the git repo is cloned | `DT_REPO` | |
| where the MIMIC-IV files are downloaded | `DT_MIMIC_DIR` | `mimic-iv` |
| where results are recorded | `DT_RESULTS_DIR` | |
| where the Ollama models are downloaded | `DT_OLLAMA_MODELS` (exported as `OLLAMA_MODELS`) | `ollama-models` |

plus the Slurm account (`DT_ACCOUNT`, exported as `SBATCH_ACCOUNT` and
`SALLOC_ACCOUNT`), the modules to load (`DT_MODULES`), the ollama binary's
directory (`DT_OLLAMA_BIN`) and a few more (see the file's header).

- **[`jobs/setup_bash.sh`](../jobs/setup_bash.sh)** is the only place
  these are set, in one block per cluster picked by `$CC_CLUSTER`. Moving to
  another cluster, or handing the project to another user, means editing
  only this file. Sourced, it sets the variables; run with `bash`, it also
  creates the `mimic-iv` and `ollama-models` symlinks and builds `.venv`.
- **`jobs/load_profile.sh`** is what each job sources. It sources
  `setup_bash.sh`, takes `--profile <file>` out of the job's arguments, and
  reads your PhysioNet credentials from **your profile**:

  ```
  ~/.config/dt_profile.yml          (default; may be missing unless you run prep1)
  --profile <file>                  (any job, to use another one)
  ```

  The profile holds only those credentials
  ([`config/profile.sample.yml`](../config/profile.sample.yml)); a `paths:`
  section left in an older profile is ignored, with a note in the job's
  output.

`DT_REPO` is the one location you never edit: `setup_bash.sh` finds it from
its own path (`${BASH_SOURCE[0]}/..`), so it is right wherever the repo is
cloned.

---

## How a job uses it

Every job script has these lines right after `set -euo pipefail`:

```bash
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the repository base ... ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"
```

Line by line:

1. **`source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh"`** — finds
   the loader. A batch job can't use its own location to find it, because
   `sbatch` copies the script to a spool directory on the compute node
   before running it. So it uses `$SLURM_SUBMIT_DIR`, the directory you ran
   `sbatch` from (or `$PWD` when the script is run by hand with `bash`).
   This is why jobs must be **submitted from the repository base**
   (`cd "$DT_REPO"`).
2. **`load_profile "$@" || exit 1`** — sets the variables (below). If
   anything is wrong it prints a message and returns 1, and `|| exit 1`
   ends the job.
3. **`set -- "${JOB_ARGS[@]}"`** — replaces the job's arguments with what's
   left after `--profile` was taken out. In
   `sbatch jobs/prep1_download_mimic_nibi.sh --profile ~/x.yml 3.1`, the
   job's `$1` afterwards is `3.1`.

The job scripts carry no `#SBATCH --account`: `sbatch` takes the account
from `SBATCH_ACCOUNT`, which `setup_bash.sh` exports. So source
`jobs/setup_bash.sh` in your shell before `sbatch` (once in `~/.bashrc` is
easiest); `jobs/run_all.sh` does it itself.

---

## What `load_profile` sets

| Variable | Comes from | Exported? | Used for |
|---|---|---|---|
| `DT_REPO`, `DT_MIMIC_DIR`, `DT_RESULTS_DIR`, `DT_OLLAMA_MODELS`, `DT_ACCOUNT`, `DT_MODULES`, ... | `jobs/setup_bash.sh` | yes | `cd "$DT_REPO"`, `module load $DT_MODULES`, `$DT_RESULTS_DIR` in the configs, prep1's download target |
| `OLLAMA_MODELS` | `$DT_OLLAMA_MODELS` | yes | the name Ollama itself reads |
| `PATH` | `$DT_OLLAMA_BIN` prepended | yes | so `ollama` is found |
| `SBATCH_ACCOUNT`, `SALLOC_ACCOUNT` | `$DT_ACCOUNT` | yes | the account of anything submitted from here |
| `PHYSIONET_USERNAME`, `PHYSIONET_PASSWORD` | profile `physionet:` | **no** | prep1 only |
| `PROFILE_FILE` | — | no | the profile that was read (empty if none), for messages |
| `JOB_ARGS` | — | no | the job's arguments minus `--profile` |

**Exported** means the variable is passed on to programs the job starts
(`python`, `ollama serve`, `wget`, ...): Python's `load_config()` expands
`$DT_RESULTS_DIR` in the configs, and `ollama` reads `$OLLAMA_MODELS`. The
PhysioNet credentials are deliberately **not** exported: they exist only in
the job's own shell, so no child process can see the password in its
environment.

`setup_bash.sh` **always overwrites** these variables, so an old
`export OLLAMA_MODELS=...` in `~/.bashrc` can't leak into a job: every
location comes from one place.

---

## `load_profile`, step by step

### 1. Defaults and cluster settings (lines 34–38)

```bash
local profile="$HOME/.config/dt_profile.yml" explicit=0 here parsed perms
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$here/jobs/setup_bash.sh" || return 1
```

- `profile` starts as the default path; `--profile` may change it below,
  and then sets `explicit=1`.
- `here` is the repo this loader lives in: `${BASH_SOURCE[0]}` is
  `$DT_REPO/jobs/load_profile.sh`, and `dirname` + `/..` goes up to
  `$DT_REPO`.
- `setup_bash.sh` sets every location for the current `$CC_CLUSTER`; on a
  cluster it has no block for, it prints `has no block for cluster ...` and
  the job stops.

### 2. Take `--profile` out of the arguments (lines 40–52)

```bash
JOB_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)   profile="$2"; explicit=1; shift 2 ;;         # --profile <file>
        --profile=*) profile="${1#--profile=}"; explicit=1; shift ;;  # --profile=<file>
        *)           JOB_ARGS+=("$1"); shift ;;                   # keep for the job
    esac
done
profile="${profile/#\~/$HOME}"
```

`--profile` can go anywhere among the arguments. Everything else is kept in
order in the `JOB_ARGS` array (so an argument containing spaces stays one
argument). The last line expands a leading `~`, which the shell doesn't do
for `--profile=~/x.yml` or a quoted `"~/x.yml"`.

### 3. Read the credentials with Python (lines 54–79)

If the profile exists, a short Python program run with `$DT_SYS_PYTHON`
(a `python3` that has PyYAML before any module or venv is loaded) parses it
with `yaml.safe_load` and prints:

```
PHYSIONET_USERNAME=<your username>
PHYSIONET_PASSWORD='<your password>'
```

each value passed through `shlex.quote`, which is what makes the following
`eval "$parsed"` safe: a password like `a'b; rm -rf ~` becomes a harmless
string, not a command. If the profile is missing, the credentials stay
empty — fine for every job except prep1, which stops with
`physionet.username is empty ...`. A missing profile named with
`--profile` is an error.

### 4. Password file permissions (lines 80–89)

```bash
perms=$(stat -L -c %a "$profile")      # e.g. 600 or 644
if (( 8#$perms & 077 )); then ...      # any group/other bits set?
```

If the profile holds a password, it must be readable only by you: any group
or "other" permission bit stops the job, with `chmod 600 <file>` as the fix.

### 5. Keep the credentials private, report (lines 94–97)

`export -n` removes the export flag from the credentials (in case an old
`export PHYSIONET_PASSWORD` is in your shell), so they stay in the job's
shell only. Then it prints the cluster, the profile and the resolved
locations — the first lines of every job's `.out` file:

```
== cluster=<cluster>  repo=$DT_REPO  profile=<profile or none> ==
==   mimic=$DT_MIMIC_DIR  results=$DT_RESULTS_DIR  ollama_models=$DT_OLLAMA_MODELS  modules=$DT_MODULES ==
```

### Why `return`, not `exit`

On any problem the function uses `return 1`; the job turns that into an
exit with `load_profile "$@" || exit 1`. In your own terminal `exit` would
close the shell, so this is safe on a login node:

```bash
source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
```

---

## Worked example

`jobs/setup_bash.sh`'s block for the cluster you are on sets, say,
`DT_MIMIC_DIR=$HOME/physionet.org/files/mimiciv/3.1`. Your profile
`~/.config/dt_profile.yml` (mode 600) holds:

```yaml
physionet:
  username: "<your username>"
  password: "my secret pass"
```

Submitted from the repository base:

```bash
source jobs/setup_bash.sh      # or once in ~/.bashrc
cd "$DT_REPO"
sbatch jobs/prep1_download_mimic_nibi.sh 3.1
```

Inside the job:

1. `load_profile 3.1` → `setup_bash.sh` sets the locations; no `--profile`,
   so the profile is `~/.config/dt_profile.yml`; `JOB_ARGS=(3.1)`.
2. Python reads the credentials; the file is mode 600 → passes the check.
3. `set -- 3.1` → the job's `$1` is `3.1` (the MIMIC-IV version), and it
   downloads into `$DT_MIMIC_DIR` — what the repo's `mimic-iv` symlink, and
   so every config's `mimic_root`, points to.

---

## Error messages

| Message (start) | Cause | Fix |
|---|---|---|
| `jobs/load_profile.sh not found — submit from the repository base` | `sbatch` was run from another directory | `cd "$DT_REPO"`, then `sbatch jobs/...` |
| `jobs/setup_bash.sh has no block for cluster '...'` | first use on a new cluster | add a block for it to `jobs/setup_bash.sh` |
| `profile '...' not found` | the file named with `--profile` doesn't exist | fix the path |
| `--profile needs a file path` | `--profile` was the last argument | put the file path after it |
| `could not read profile ...` | file unreadable or not valid YAML (often a tab, or a missing space after `:`) | fix the line the message points to |
| `holds a password but is readable by others (mode 644)` | file permissions too open | `chmod 600 <file>` |
| `physionet.username is empty in ...` (from prep1) | no profile, or credentials not filled in | `cp config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 ~/.config/dt_profile.yml`, then fill in `physionet:` |
| `paths.<key> in <config> uses an unset variable` (from Python) | `$DT_RESULTS_DIR` not set: run outside a job without sourcing | `source jobs/setup_bash.sh` |
| sbatch: `You must specify an account` or similar | `SBATCH_ACCOUNT` not set in the submitting shell | `source jobs/setup_bash.sh` |
