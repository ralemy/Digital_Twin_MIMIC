# `jobs/load_profile.sh` — how jobs get their settings

This page walks through `jobs/load_profile.sh` from the top down: first what
it is for and how a job uses it, then the `load_profile` function step by
step, then a worked example and the error messages you can run into. Line
numbers refer to [`jobs/load_profile.sh`](../jobs/load_profile.sh).

---

## What it's for

Every job in `jobs/` needs a few site-specific values: where the repo is,
where the data lives, where the Ollama models are, and (for the MIMIC-IV
download) your PhysioNet login. Instead of each job reading these from
environment variables you remember to `export`, they all come from one
private YAML file, **your profile**:

```
~/.config/dt_profile.yml          (default)
--profile <file>                  (any job, to use another one)
```

`load_profile.sh` is the one piece of code that reads that file. It is
**sourced** by the jobs (run inside their shell), never submitted itself.
It defines a single function, `load_profile`, and does nothing until a job
calls it.

The profile's format is documented in
[`config/profile.sample.yml`](../config/profile.sample.yml); setting it up
is described in the README, section "Your profile".

---

## How a job uses it

Every job script has these lines right after `set -euo pipefail`:

```bash
source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh" \
    || { echo "== jobs/load_profile.sh not found — submit from the project directory ... ==" >&2; exit 1; }
load_profile "$@" || exit 1
set -- "${JOB_ARGS[@]}"
```

Line by line:

1. **`source "${SLURM_SUBMIT_DIR:-$PWD}/jobs/load_profile.sh"`** — finds
   the loader. A batch job can't use its own location to find it, because
   `sbatch` copies the script to a spool directory on the compute node
   before running it. So it uses `$SLURM_SUBMIT_DIR`, the directory you ran
   `sbatch` from (or `$PWD` when the script is run by hand with `bash`).
   This is why jobs must be **submitted from the project directory**. If
   the file isn't there, the `|| { ...; exit 1; }` part stops the job with
   a message saying so.
2. **`load_profile "$@" || exit 1`** — calls the function with *all* the
   arguments the job received. The function reads the profile and sets the
   variables (below). If anything is wrong it prints a message and returns
   1, and `|| exit 1` ends the job.
3. **`set -- "${JOB_ARGS[@]}"`** — replaces the job's arguments (`$1`,
   `$2`, ...) with what's left after `--profile` was taken out, so the rest
   of the job sees exactly the arguments it always did. For example, in
   `sbatch jobs/step1_resolve_items_nibi.sh --profile ~/x.yml config/config_nibi_lean.yaml`,
   the job's `$1` afterwards is `config/config_nibi_lean.yaml`.

---

## What `load_profile` sets

| Variable | From profile key | Default if key missing | Exported? | Used for |
|---|---|---|---|---|
| `AGENTIC_DT_PRJ` | `paths.project_dir` | the repo `load_profile.sh` is in | yes | `cd` into the repo, find `.venv`, `src/`, `config/` |
| `PROJECT` | `paths.data_root` | `project_dir` | yes | the `$PROJECT` in `config_nibi_*.yaml` paths; prep1 and verify_mimic's MIMIC-IV location |
| `OLLAMA_MODELS` | `paths.ollama_models` | **required — no default** | yes | Ollama's model store (the name Ollama itself reads) |
| `PATH` | `paths.ollama_bin` prepended | `~/ollama-local/bin` | yes | so `ollama` is found |
| `PHYSIONET_USERNAME` | `physionet.username` | empty | **no** | prep1 only |
| `PHYSIONET_PASSWORD` | `physionet.password` | empty | **no** | prep1 only |
| `PROFILE_FILE` | — | — | no | the profile that was read, for messages |
| `JOB_ARGS` | — | — | no | the job's arguments minus `--profile` |

**Exported** means the variable is passed on to programs the job starts
(`python`, `ollama serve`, `wget`, ...). The first four must be: Python's
`load_config()` expands `$PROJECT` in the config, and `ollama` reads
`$OLLAMA_MODELS`. The PhysioNet credentials are deliberately **not**
exported: they exist only in the job's own shell, so no child process can
see the password in its environment.

The profile **always wins** over variables already in the environment. If
your `~/.bashrc` still has `export OLLAMA_MODELS=...`, `sbatch` passes it
into the job, but `load_profile` overwrites it with the profile's value.
This keeps every job's settings coming from one place.

---

## `load_profile`, step by step

### 1. Defaults (lines 33–34)

```bash
local profile="$HOME/.config/dt_profile.yml" here parsed perms
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

- `profile` starts as the default path; `--profile` may change it below.
- `here` is the repo this loader lives in. `${BASH_SOURCE[0]}` is the path
  of the file currently being sourced (`.../exp1/jobs/load_profile.sh`);
  `dirname` gives `.../exp1/jobs`, and `/..` goes up one level to
  `.../exp1`. It is used as the default `project_dir` and for the sanity
  check in step 5.
- `local` keeps these four helper variables inside the function, so they
  don't leak into the job.

### 2. Take `--profile` out of the arguments (lines 36–48)

```bash
JOB_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)   profile="$2"; shift 2 ;;     # --profile <file>
        --profile=*) profile="${1#--profile=}"; shift ;;   # --profile=<file>
        *)           JOB_ARGS+=("$1"); shift ;;   # anything else: keep for the job
    esac
done
profile="${profile/#\~/$HOME}"
```

The loop looks at the arguments one at a time (`$1`) and `shift`s each one
off when done:

- `--profile <file>` uses the next argument as the profile and skips both.
  If `--profile` is the last argument (no file after it), it stops with
  `--profile needs a file path`.
- `--profile=<file>` does the same in one argument. `${1#--profile=}`
  strips the `--profile=` prefix, leaving the path.
- Everything else is appended to the `JOB_ARGS` array, in order. Because
  it's an array, an argument containing spaces stays one argument.

So `--profile` can go anywhere: before, between or after the job's own
arguments.

The last line expands a leading `~` to your home directory
(`${var/#pattern/replacement}` replaces `pattern` only at the start). The
shell normally does that before the job sees the argument, but not for
`--profile=~/x.yml` or a quoted `"~/x.yml"`, so this covers those cases.

### 3. Make sure the profile exists (lines 50–55)

If the file isn't there, the function prints the two commands that create
it from the sample and returns 1, so the job stops before doing anything.

### 4. Read the YAML with Python (lines 57–89)

```bash
parsed=$(/usr/bin/python3 - "$profile" "$here" <<'PY'
...python...
PY
) || return 1
eval "$parsed"
```

Bash can't read YAML, so a short Python program does it. The pieces:

- **`/usr/bin/python3`, not `python`** — the system Python, which already
  has PyYAML. The job hasn't loaded `module load python/3.11` or the venv
  yet at this point (and can't: the venv's location comes *from* the
  profile), and the module's Python doesn't include PyYAML.
- **`python3 - "$profile" "$here" <<'PY' ... PY`** — `-` tells Python to
  read the program from standard input, which is the text between
  `<<'PY'` and `PY` (a "here-document"). The quotes in `'PY'` stop bash
  from touching `$` signs inside the Python code. The two arguments arrive
  in Python as `sys.argv[1]` and `sys.argv[2]`.

What the Python does (lines 58–86):

1. Opens and parses the file with `yaml.safe_load` (which only builds
   plain data — it can't run code hidden in the YAML). A missing file or
   broken YAML ends it with `could not read profile ...`.
2. Takes the `paths:` and `physionet:` sections, treating a missing
   section as empty.
3. Requires `paths.ollama_models`; every other key has a default (see the
   table above).
4. Expands each path with `expand()`: `os.path.expandvars` replaces
   `$SCRATCH`, `$HOME`, etc., and `os.path.expanduser` replaces a leading
   `~`.
5. Prints one `NAME=value` line per variable, with the value passed
   through `shlex.quote`, e.g.:

   ```
   AGENTIC_DT_PRJ=/home/ralemy/projects/def-roudsari/digital_twin/exp1
   PROJECT=/home/ralemy/projects/def-roudsari/digital_twin/exp1
   OLLAMA_MODELS=/home/ralemy/projects/def-roudsari/ollama-local/models
   DT_OLLAMA_BIN=/home/ralemy/ollama-local/bin
   PHYSIONET_USERNAME=ralemy
   PHYSIONET_PASSWORD='my secret pass'
   ```

Bash captures that output in `parsed` (the `$( ... )`). If Python exited
with an error, `|| return 1` stops here; Python's own message has already
been printed. Otherwise `eval "$parsed"` runs those lines as bash
assignments, which sets the variables.

`shlex.quote` is what makes the `eval` safe: it wraps any value that
contains spaces, quotes, `$`, `;` and so on in single quotes, so bash
treats it as plain text. A password like `a'b; rm -rf ~` becomes a
harmless string, not a command.

### 5. Safety checks (lines 91–107)

**Password file permissions.** If the profile contains a password, the
file must be readable only by you:

```bash
perms=$(stat -L -c %a "$profile")      # e.g. 600 or 644
if (( 8#$perms & 077 )); then ...      # any group/other bits set?
```

`stat -c %a` prints the permission bits as an octal number like `644`
(`-L` follows a symlink to the real file). `8#$perms` tells bash to read
it as octal, and `& 077` keeps only the group and "other" bits. Anything
non-zero means someone besides you can read or write the file, so the job
clears the password from memory and stops with `chmod 600 <file>` as the
fix. A profile *without* a password is allowed to be readable, since it
holds only paths.

**`project_dir` must exist.** A typo in the path is caught here, not later
as a confusing `cd` or `.venv` error.

**`project_dir` vs. the submit directory.** `[ a -ef b ]` is true when `a`
and `b` are the same directory, even through symlinks (e.g.
`~/projects/def-roudsari/...` and `/project/6116810/...`). If they differ,
the job runs the loader from one checkout but the code and venv from
another. That's allowed (it might be what you want), so it is only a
`WARNING` line in the output, not an error.

### 6. Export and report (lines 109–118)

```bash
case ":$PATH:" in
    *":$DT_OLLAMA_BIN:"*) ;;                      # already on PATH
    *) PATH="$DT_OLLAMA_BIN:$PATH" ;;             # otherwise put it first
esac
export AGENTIC_DT_PRJ PROJECT OLLAMA_MODELS PATH
export -n PHYSIONET_USERNAME PHYSIONET_PASSWORD 2>/dev/null || true
```

- The `case` adds the Ollama directory to `PATH` only once. Wrapping both
  sides in `:` makes the match exact, so `/a/bin` doesn't match
  `/a/bin2`.
- `export` passes the four non-secret variables to child processes.
- `export -n` does the opposite for the credentials: if they were
  exported before (say, from an old `export PHYSIONET_PASSWORD` in your
  shell), it removes that, so they stay private to the job's shell.
- Finally it prints which profile it used and the resolved paths. These
  lines appear at the top of every job's `.out` file, so you can always
  see what a run used.

### Why `return`, not `exit`

On any problem the function uses `return 1`, not `exit 1`. The job turns
that into an exit with `load_profile "$@" || exit 1`. The difference
matters when you use the loader in your own terminal: `exit` would close
your shell, while `return` just reports the error. So this is safe on a
login node:

```bash
source jobs/load_profile.sh && load_profile && echo "$OLLAMA_MODELS"
```

---

## Worked example

Profile `~/.config/dt_profile.yml`, mode 600:

```yaml
paths:
  data_root: $SCRATCH/dt-data
  ollama_models: ~/projects/def-roudsari/ollama-local/models
physionet:
  username: ralemy
  password: "my secret pass"
```

Submitted from the project directory:

```bash
cd /home/ralemy/projects/def-roudsari/digital_twin/exp1
sbatch jobs/prep1_download_mimic_nibi.sh 3.1
```

Inside the job:

1. `load_profile 3.1` → no `--profile`, so the profile is
   `~/.config/dt_profile.yml`; `JOB_ARGS=(3.1)`.
2. Python reads the file. `project_dir` is missing, so it defaults to the
   repo; `data_root` expands `$SCRATCH` to `/scratch/ralemy/dt-data`;
   `ollama_bin` defaults to `~/ollama-local/bin`.
3. The file has a password and mode 600 → passes the check.
4. Variables set: `PROJECT=/scratch/ralemy/dt-data`, etc. The password is
   in the job's shell but not exported.
5. `set -- 3.1` → the job's `$1` is `3.1` (the MIMIC-IV version), and it
   downloads into `$PROJECT/mimic-iv` = `/scratch/ralemy/dt-data/mimic-iv`.

Same job with another profile, `--profile` placed after the version:

```bash
sbatch jobs/prep1_download_mimic_nibi.sh 3.1 --profile ~/.config/dt_profile_test.yml
```

`JOB_ARGS` is still `(3.1)`; only the profile file changes.

---

## Error messages

| Message (start) | Cause | Fix |
|---|---|---|
| `jobs/load_profile.sh not found — submit from the project directory` | `sbatch` was run from another directory | `cd` to the repo, then `sbatch jobs/...` |
| `profile '...' not found` | no profile at the default path or the `--profile` path | `cp config/profile.sample.yml ~/.config/dt_profile.yml && chmod 600 ~/.config/dt_profile.yml`, then edit it |
| `--profile needs a file path` | `--profile` was the last argument | put the file path after it |
| `could not read profile ...` | file unreadable or not valid YAML (often a tab, or a missing space after `:`) | fix the line the message points to |
| `profile ... has no paths.ollama_models` | key missing or empty | add it under `paths:` |
| `holds a password but is readable by others (mode 644)` | file permissions too open | `chmod 600 <file>` |
| `paths.project_dir '...' does not exist` | wrong path in the profile | correct it, or remove the key to use the repo you submit from |
| `WARNING: paths.project_dir (...) is not the repo this job was submitted from` | profile points at a different checkout | fine if intended; otherwise fix `project_dir` or submit from that checkout |
| `physionet.username is empty in ...` (from prep1) | credentials not filled in | fill in `physionet:` in the profile |
