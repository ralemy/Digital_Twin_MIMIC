# How jobs get their settings: your profile, `jobs/setup_bash.sh` and `jobs/load_profile.sh`

This page walks through how every job in `jobs/` learns where things are:
first what each piece is for, then how a job uses them, then
`setup_bash.sh` and `load_profile` step by step, a worked example and the
error messages you can run into. Line numbers refer to
[`jobs/setup_bash.sh`](../jobs/setup_bash.sh) and
[`jobs/load_profile.sh`](../jobs/load_profile.sh).

---

## What they're for

Every job needs a few values that differ between Alliance clusters (Nibi,
Rorqual, ...) and between users:

| Location / setting | Profile key | Variable | Repo symlink |
|---|---|---|---|
| where the git repo is cloned | — (found automatically) | `DT_REPO` | |
| where the MIMIC-IV files are downloaded | `paths.mimic_dir` | `DT_MIMIC_DIR` | `mimic-iv` |
| where results are recorded | `paths.results_dir` | `DT_RESULTS_DIR` | |
| where the Ollama models are downloaded | `paths.ollama_models` | `DT_OLLAMA_MODELS` (also `OLLAMA_MODELS`) | `ollama-models` |
| the ollama binary's directory | `paths.ollama_bin` | `DT_OLLAMA_BIN` (first on `PATH`) | |
| the Slurm account | `slurm.account` | `DT_ACCOUNT` (also `SBATCH_ACCOUNT`, `SALLOC_ACCOUNT`) | |
| modules to load | `environment.modules` | `DT_MODULES` | |
| keep Weights & Biases off (default `true`) | `environment.disable_wandb` | `DT_WANDB_DISABLED` (also `WANDB_MODE=disabled`) | |
| compute nodes reach the internet (default `true`) | `environment.workers_have_internet` | `DT_WORKER_INTERNET` | |
| PhysioNet login (download job only) | `physionet.username`, `.password` | not exported | |

- **Your profile, `~/.config/dt_profile.yml`** (template:
  [`config/profile.sample.yml`](../config/profile.sample.yml)), is the only
  file you edit. Alliance clusters don't share home directories, so there is
  one per cluster. It must be mode 600: it holds your PhysioNet password.
- **`jobs/setup_bash.sh`** reads the profile and sets the variables.
  Sourced, it only sets them (this is what jobs do); run once with `bash`
  — the setup — it also creates the `mimic-iv` and `ollama-models`
  symlinks, installs Ollama, builds `.venv` and writes the final values into
  `~/.bashrc`, logging to `setup.lock`. It does not read the PhysioNet
  credentials.
- **`jobs/load_profile.sh`** is what each job sources. It takes
  `--profile <file>` out of the job's arguments, sources `setup_bash.sh`
  with that profile, and reads the PhysioNet credentials.

`DT_REPO` is the one location never configured: `setup_bash.sh` finds it
from its own path (`${BASH_SOURCE[0]}/..`), so it is right wherever the
repo is cloned. Profile paths may use it (`$DT_REPO/...`).

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
from `SBATCH_ACCOUNT`, which `setup_bash.sh` exports from `slurm.account`.
The setup writes it into `~/.bashrc`, so every login shell has it after
`source ~/.bashrc`; `jobs/run_all.sh` reads the profile itself.

---

## `jobs/setup_bash.sh`, step by step

### 1. Where things are (lines 67–77)

- `DT_REPO` is the directory above `jobs/`.
- `DT_CLUSTER` is `$CC_CLUSTER` (set by the Alliance environment), or
  `local`.
- `DT_PROFILE` is `$DT_PROFILE` if already set (by `load_profile`'s
  `--profile`, or by you), else `~/.config/dt_profile.yml`.
- `DT_SYS_PYTHON` is `/usr/bin/python3` if it has PyYAML (true on Alliance
  clusters), else the first `python3` on `PATH`. It must work before any
  module or `.venv` is loaded.

### 2. Read the profile (`dt_read_profile`, lines 107–193)

1. **It must exist** — otherwise it prints the `cp` and `chmod` commands
   that create it from the template.
2. **It must be private.** `stat -c %a` prints the permission bits (e.g.
   `600` or `644`); `8#$perms & 077` keeps the group and "other" bits.
   Anything non-zero means someone else can read the file, so it stops with
   `chmod 600 <file>` as the fix.
3. **A short Python program parses it** with `yaml.safe_load` (which can't
   run code hidden in the YAML) and checks it:
   - `paths.mimic_dir`, `paths.results_dir` and `paths.ollama_models` are
     required; `paths.ollama_bin` defaults to `~/ollama-local/bin`.
   - Each path has `~` and environment variables (`$HOME`, `$SCRATCH`,
     `$USER`, `$DT_REPO`, ...) expanded and must then be absolute; a variable
     that isn't set is an error.
   - On a cluster, `slurm.account` is required, and `environment.modules`
     defaults to `StdEnv/2023 python/3.11`.
   - `environment.disable_wandb` and `environment.workers_have_internet`
     must be true or false (default true); they become `1` or `0`. With
     `DT_WANDB_DISABLED=1`, `src/tracking.py` never starts Weights & Biases;
     with `DT_WORKER_INTERNET=0`, the download jobs refuse to run as Slurm
     jobs and `run_all.sh` runs its `models` stage on the login node.
   - Every problem is listed at once (`paths.mimic_dir is empty`, ...).
4. It prints one `NAME=value` line per variable, each value passed through
   `shlex.quote`, and `eval` turns them into shell variables. The quoting
   is what makes `eval` safe: any `$`, quote or `;` in a value stays text.

If anything fails, a sourced `setup_bash.sh` returns 1 (an executed one
exits 1), so a job stops right there.

### 3. Export (lines 202–222)

`OLLAMA_MODELS` (what Ollama reads) is set to `DT_OLLAMA_MODELS`;
`DT_OLLAMA_BIN` is put first on `PATH` (once); `SBATCH_ACCOUNT` and
`SALLOC_ACCOUNT` are set to `DT_ACCOUNT`; and everything is exported, so
`python`, `ollama serve` and `sbatch` see it. The profile always wins: an
old `export OLLAMA_MODELS=...` in `~/.bashrc` is overwritten.

### 4. One-time setup (only with `bash jobs/setup_bash.sh`)

- **Lock and log (lines 80–103, before the profile is read):** creates
  `setup.lock` in the repo base atomically (`noclobber`) — or refuses to run
  if it exists — and copies everything printed into it (`exec > >(tee ...)`).
  An `EXIT` trap renames it to `setup-failed-<time>.log` if the run fails,
  so a failed setup never blocks the next one.
- **Symlinks:** `$DT_REPO/mimic-iv` → `$DT_MIMIC_DIR` and
  `$DT_REPO/ollama-models` → `$DT_OLLAMA_MODELS`, creating the targets (and
  `$DT_RESULTS_DIR`) if needed. A real directory in place of a link is an
  error, never overwritten.
- **Ollama:** if `$DT_OLLAMA_BIN/ollama` doesn't exist, downloads release
  `$DT_OLLAMA_VERSION` from GitHub and unpacks it into the parent of
  `$DT_OLLAMA_BIN` (no root needed).
- **`.venv`**, if it doesn't exist: `module load $DT_MODULES`,
  `virtualenv --no-download`, `pip install --no-index -r requirements.txt`.
- **`~/.bashrc`:** replaces (or appends) the block between
  `# >>> mimic-iv digital twin ...` and `# <<< mimic-iv digital twin <<<`
  with `export` lines holding the final values (each quoted with
  `printf %q`), `SBATCH_ACCOUNT`/`SALLOC_ACCOUNT`, and a `PATH` line for
  the ollama binary. The previous file is kept as `~/.bashrc.dt-backup`;
  other lines setting the same variables are listed, not changed.

---

## `load_profile`, step by step

### 1. Take `--profile` out of the arguments (lines 37–48)

```bash
JOB_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)   DT_PROFILE="$2"; shift 2 ;;              # --profile <file>
        --profile=*) DT_PROFILE="${1#--profile=}"; shift ;;   # --profile=<file>
        *)           JOB_ARGS+=("$1"); shift ;;               # keep for the job
    esac
done
```

`--profile` can go anywhere among the arguments. Everything else is kept in
order in the `JOB_ARGS` array (so an argument containing spaces stays one
argument).

### 2. Source `setup_bash.sh` (line 51)

With `DT_PROFILE` set from `--profile` (or inherited, or the default), so
the locations and the credentials come from the same file.

### 3. Read the credentials (lines 53–66)

The same kind of Python program reads `physionet.username` and
`physionet.password` (empty if not filled in — fine for every job except
prep1, which stops with `physionet.username is empty in ...`). `export -n`
then makes sure they are **not** exported, even if an old
`export PHYSIONET_PASSWORD` is in your shell: they exist only in the job's
own shell, so no child process can see the password in its environment.

### 4. Report (lines 68–69)

The first lines of every job's `.out` file:

```
== cluster=<cluster>  account=<account>  repo=$DT_REPO  profile=<profile> ==
==   mimic=$DT_MIMIC_DIR  results=$DT_RESULTS_DIR  ollama_models=$DT_OLLAMA_MODELS  modules=$DT_MODULES ==
```

### Why `return`, not `exit`

On any problem the functions use `return 1`; the job turns that into an
exit with `load_profile "$@" || exit 1`. In your own terminal `exit` would
close the shell, so this is safe on a login node:

```bash
source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
```

---

## Worked example

Your profile `~/.config/dt_profile.yml` (mode 600) holds:

```yaml
slurm:
  account: def-yourpi
paths:
  mimic_dir: ~/projects/def-yourpi/mimic-iv/3.1
  results_dir: ~/projects/def-yourpi/dt-results
  ollama_models: $SCRATCH/ollama-local/models
physionet:
  username: "<your username>"
  password: "my secret pass"
```

Submitted from the repository base:

```bash
source ~/.bashrc               # after the setup — sets SBATCH_ACCOUNT=def-yourpi
cd "$DT_REPO"
sbatch jobs/prep1_download_mimic_nibi.sh 3.1
```

Inside the job:

1. `load_profile 3.1` → no `--profile`, so `DT_PROFILE` is
   `~/.config/dt_profile.yml`; `JOB_ARGS=(3.1)`.
2. `setup_bash.sh` checks the mode (600, fine) and sets
   `DT_MIMIC_DIR=/home/<you>/projects/def-yourpi/mimic-iv/3.1`,
   `DT_RESULTS_DIR=...`, `OLLAMA_MODELS=/scratch/<you>/ollama-local/models`,
   `DT_OLLAMA_BIN=/home/<you>/ollama-local/bin` (the default).
3. The credentials are read into the job's shell, not exported.
4. `set -- 3.1` → the job's `$1` is `3.1` (the MIMIC-IV version), and it
   downloads into `$DT_MIMIC_DIR` — what the repo's `mimic-iv` symlink, and
   so every config's `mimic_root`, points to.

---

## Error messages

| Message (start) | Cause | Fix |
|---|---|---|
| `jobs/load_profile.sh not found — submit from the repository base` | `sbatch` was run from another directory | `cd "$DT_REPO"`, then `sbatch jobs/...` |
| `no profile at ...` | no profile yet (or a wrong `--profile` path) | `cp config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 ~/.config/dt_profile.yml`, then fill it in |
| `profile ... is readable by others (mode 644)` | file permissions too open | `chmod 600 <file>` |
| `setup has already run in this clone` | `setup.lock` exists | to redo the setup: `rm setup.lock && bash jobs/setup_bash.sh` |
| `setup FAILED` | the step above it failed (e.g. the Ollama download) | fix it and run the setup again — no lock is left |
| `profile ... is incomplete` + a list | required keys empty, a path not absolute, or an unset variable in a path | fill in the keys listed |
| `(it uses the old format ...)` | a profile from before `mimic_dir`/`results_dir` existed | start again from `config/profile.sample.yml` |
| `could not read profile ...` | not valid YAML (often a tab, or a missing space after `:`) | fix the line the message points to |
| `--profile needs a file path` | `--profile` was the last argument | put the file path after it |
| `physionet.username is empty in ...` (from prep1) | credentials not filled in | fill in `physionet:` in the profile |
| `paths.<key> in <config> uses an unset variable` (from Python) | `$DT_RESULTS_DIR` not set: run outside a job in a shell without the setup's variables | `source ~/.bashrc` (after the setup) |
| sbatch: `You must specify an account` or similar | `SBATCH_ACCOUNT` not set in the submitting shell | `source ~/.bashrc` (after the setup) |
