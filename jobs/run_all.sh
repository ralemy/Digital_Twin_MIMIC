#!/bin/bash
# =============================================================================
# The whole pipeline for one scope, as Slurm jobs — run on a LOGIN NODE:
#   bash jobs/run_all.sh [lean]                  # 5-variable panel (the default scope)
#   bash jobs/run_all.sh full                    # 19-variable panel
#   bash jobs/run_all.sh lean --plan             # show the jobs it would submit
#   bash jobs/run_all.sh lean --status           # what's done / running
#   bash jobs/run_all.sh lean --stop             # stop the driver (not the jobs)
#   bash jobs/run_all.sh lean --run-name funny_rabbit   # resume / name a run
#   bash jobs/run_all.sh full --until tune       # stop after a stage (run again without it to go on)
# Other options: -i <seconds> (how often Slurm is checked, at least 60,
# default 120), --foreground (don't detach), --redo-extract (re-run steps 1-2
# even if their outputs exist), --profile <file> (passed on to every job),
# --unattended (see "Unattended runs" below), --redo <stage,...> (run
# finished stages again, e.g. after adding conditions: --redo calibrate,run,
# evaluate; their checkpoints mean only new work is computed), --until
# <stage> (stop once that stage is done, e.g. to measure a scope's tuning
# before committing GPU time to the rest; the same command without --until
# then carries on from the next stage).
#
# Run names. Every run has a name (--run-name; default: a new random one
# such as sassy_hammer, printed when the driver starts). All run_all calls
# with the same name share:
#   tuned_configs/<run>/                         the run's copy of the config
#                                                and its tuned config
#   $DT_RESULTS_DIR/mimic-iv-twin-work[-full]/<run>/   work, cache, results
#   $DT_LOGS_ROOT/<run>/                         driver log, job .out files,
#                                                Ollama logs
#   run_all/<run>/                               the driver's state files
# When every stage is done, the driver packs them (jobs/pack_run.sh) into
# $DT_RESULTS_DIR/<run>_results.tar. So a new name starts from scratch
# (resolve and extract run again), and the same name resumes. --status, --stop and --plan without --run-name use the
# scope's latest run (run_all/<scope>.latest).
#
# Stages, each run by the job script already in jobs/:
#   models     prep2_download_models.sh        <config>   (pulls only what's missing)
#   resolve    step1_resolve_items.sh     <config>
#   extract    step2_extract_cohort.sh    <config>
#   tune       step3b_tune.sh             <config>   -> <config>_tuned.yaml (same directory)
#   calibrate  step3c_calibrate.sh        <tuned config>
#   run        step3_run_experiment.sh    <tuned config>
#   evaluate   step4_evaluate_results.sh  <tuned config>
# Jobs are submitted through jobs/submit.sh, which adapts them to the cluster
# (on Trillium: run this from the GPU login node; every job takes one GPU).
# lean = config/config_alliance_lean.yaml, full = config/config_alliance_full.yaml;
# <config> is the run's copy, tuned_configs/<run>/<config file>, rewritten
# from it each time a driver starts.
# evaluate writes statistical_analysis.json (RQ1-RQ3) to the tuned results_dir.
# resolve/extract are skipped if their outputs already exist. After resolve
# runs, the driver stops so item_mapping.json can be reviewed; run the same
# command again to continue.
#
# Time limits. Each stage's estimated work (EST_* below, from
# docs/runtime_estimates.md, upper ends of the ranges) is split into jobs of
# at most 7 h of work, each with a time limit 1 h longer than its work,
# capped at 8 h. The jobs of a stage are submitted together as a chain: job
# k+1 depends on afternotok of jobs 1..k, so it starts only if the job
# before it ran out of time or failed, and resumes from the checkpoints the
# Python step writes per batch. Once a job COMPLETES, Slurm cancels the rest
# of its chain (--kill-on-invalid-dep). If a chain runs out of time without
# finishing, the driver submits one more job (the stage's estimate, at most
# 7 h of work, so at most 8 h), up to MAX_ROUNDS times.
# MAX_FAILS consecutive failed (not timed-out) jobs stop the driver.
#
# Unattended runs (--unattended), e.g. a whole lean run overnight:
#   bash jobs/run_all.sh lean --unattended
# A stage that stops on failures (failed jobs, a failed sbatch, Slurm
# hiccups) is resubmitted from its checkpoints after a cooldown (15 min,
# doubling up to 2 h), up to MAX_STAGE_RETRIES times; time-outs get up to
# MAX_ROUNDS_UNATTENDED extra jobs; and the driver doesn't pause after
# resolve for the item-mapping review (review it afterwards). Only a job
# cancelled by you or an administrator, or retries running out, stops it.
#
# Resuming. Stage status and job ids are kept in run_all/<run>/<scope>.state.
# The driver detaches from the terminal (setsid + nohup) and logs to
# $DT_LOGS_ROOT/<run>/run_all-<scope>.log, which this command then follows:
# Ctrl+C or a dropped SSH connection stops only the following. Running the
# same command with the same --run-name again re-attaches to a live driver, or, if it has died (login node rebooted,
# stopped), starts a new one that picks up the jobs already submitted. A
# stage that failed is resubmitted from its checkpoints. Jobs run under
# Slurm either way: a dead driver only delays submitting the NEXT stage.
#
# Cluster etiquette (Alliance): Slurm is queried once per interval (at least
# 60 s) through jobs/monitor-job.sh, one driver runs per scope, it refuses to
# run inside a job, and it is idle between checks. (One driver per scope and
# run name: different run names run side by side.) The Alliance docs also
# suggest tmux for long sessions; --foreground inside tmux works as well.
# =============================================================================
set -uo pipefail

INTERVAL=120
MAX_JOB_MIN=480                              # 8 h per job at most
MARGIN_MIN=60                                # each job gets >= 1 h more than its planned work
WORK_PER_JOB=$((MAX_JOB_MIN - MARGIN_MIN))   # so a full-length job plans 7 h of work
MAX_ROUNDS=3                                 # extra 8 h jobs after a stage's chain ran out of time
MAX_FAILS=2                                  # consecutive failed jobs before giving up
MAX_ROUNDS_UNATTENDED=6                      # --unattended: extra jobs after time-outs
MAX_STAGE_RETRIES=5                          # --unattended: resubmissions of a failed stage
COOLDOWN_MIN=${RUN_ALL_COOLDOWN_MIN:-15}     # --unattended: first wait before a resubmission (doubles, max 120)

STAGES=(models resolve extract tune calibrate run evaluate)
declare -A SCRIPT=(
    [models]=jobs/prep2_download_models.sh
    [resolve]=jobs/step1_resolve_items.sh
    [extract]=jobs/step2_extract_cohort.sh
    [tune]=jobs/step3b_tune.sh
    [calibrate]=jobs/step3c_calibrate.sh
    [run]=jobs/step3_run_experiment.sh
    [evaluate]=jobs/step4_evaluate_results.sh
)
# Estimated work in minutes (docs/runtime_estimates.md), for the 16 LLM
# conditions in the configs at OLLAMA_NUM_PARALLEL=8 with flash attention
# (performance.llm_max_concurrent_requests: 8, ollama_flash_attention: true).
# Lean, measured on Trillium (tri_lean_exp2, H100, per condition on the 450
# test patients): Qwen2.5-32B 29-36 min, Baichuan-M2 37-38, Gemma 3 /
# MedGemma 40-50, MedGemma + Gemma 3 critic 66, Llama-3-70B / Med42-70B
# 55-58. With Med42 as llm.model its 4 primary conditions run ~1.55x longer
# than Qwen's did: run ~780 min of LLM work (tri_lean_exp2: 728, 12.3 h
# wall), calibration (322 patients) ~580 (550, 9.3 h), tuning 20 Med42
# evaluations of ~15 min on 128 patients plus ~30 min of baselines and model
# loads (tri_lean_exp2: 26 Qwen evaluations of 9.7 min, 4.3 h). full: not
# yet measured on Trillium; 3.7x the lean LLM time (19-variable answers),
# from the Nibi pilot. models: ~78 GB to download at the ~24 MB/s job
# 23132444 measured; near zero once everything is present (2 min there).
declare -A EST_LEAN=([models]=90 [resolve]=5 [extract]=10 [tune]=330 [calibrate]=600  [run]=800  [evaluate]=30)
declare -A EST_FULL=([models]=90 [resolve]=5 [extract]=20 [tune]=1140 [calibrate]=2160 [run]=2900 [evaluate]=30)

usage() { sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

ORIG_ARGS=("$@")
SCOPE=""; ACTION=run; FOREGROUND=0; REDO_EXTRACT=0; UNATTENDED=0; REDO=""; RUN_NAME=""; UNTIL=""
PROFILE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        lean|full) SCOPE=$1; shift ;;
        -i|--interval) INTERVAL="${2:-}"; shift 2 ;;
        --foreground) FOREGROUND=1; shift ;;
        --redo-extract) REDO_EXTRACT=1; shift ;;
        --unattended) UNATTENDED=1; shift ;;
        --redo) REDO="${2:-}"; shift 2 ;;
        --redo=*) REDO="${1#--redo=}"; shift ;;
        --until) UNTIL="${2:-}"; shift 2 ;;
        --until=*) UNTIL="${1#--until=}"; shift ;;
        --plan) ACTION=plan; shift ;;
        --status) ACTION=status; shift ;;
        --stop) ACTION=stop; shift ;;
        --run-name) RUN_NAME="${2:-}"; shift 2 ;;
        --run-name=*) RUN_NAME="${1#--run-name=}"; shift ;;
        --profile) PROFILE_ARGS+=("$1" "${2:-}"); shift 2 ;;
        --profile=*) PROFILE_ARGS+=("$1"); shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done
SCOPE=${SCOPE:-lean}                # config/config_alliance_lean.yaml unless "full" is given
if [ -n "$UNTIL" ] && [[ " ${STAGES[*]} " != *" $UNTIL "* ]]; then
    echo "== --until: unknown stage '$UNTIL' (stages: ${STAGES[*]}) ==" >&2
    exit 1
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 60 ]; then
    echo "== interval must be a whole number of seconds, at least 60 (Alliance: don't poll Slurm more often) ==" >&2
    exit 1
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "== run this on a login node, not inside a Slurm job (SLURM_JOB_ID=$SLURM_JOB_ID) ==" >&2
    exit 1
fi

source "$(cd "$(dirname "$0")" && pwd)/load_profile.sh" || exit 1
load_profile "${PROFILE_ARGS[@]}" || exit 1
cd "$DT_REPO" || exit 1

if [ "$SCOPE" = lean ]; then
    BASE_CONFIG=config/config_alliance_lean.yaml
    declare -n EST=EST_LEAN
else
    BASE_CONFIG=config/config_alliance_full.yaml
    declare -n EST=EST_FULL
fi

# --- run name --------------------------------------------------------------------
ADJECTIVES=(agile bold brave breezy bright calm cheeky clever cosmic crafty curious dapper
            eager fancy fearless fuzzy gentle giddy groovy happy humble jolly jumpy keen
            lively lucky mellow merry mighty nimble noble peppy plucky quirky quick rapid
            sassy shiny silly sleepy snappy spicy sunny swift tidy witty zany zesty)
NOUNS=(badger beacon biscuit cactus comet falcon ferret gecko hammer harbor kettle koala
       lantern lemur llama mango marble meadow narwhal noodle otter panda pancake pebble
       pelican penguin pickle puffin quokka rabbit raccoon rocket sparrow squid teapot
       tiger toucan tulip turnip walrus wombat yak zebra)
run_name_taken() {
    [ -e "run_all/$1" ] || [ -e "$DT_LOGS_ROOT/$1" ] || [ -e "tuned_configs/$1" ] ||
        compgen -G "$DT_RESULTS_DIR/mimic-iv-twin-work*/$1" > /dev/null
}
new_run_name() {
    local name i
    for i in {1..50}; do
        name="${ADJECTIVES[RANDOM % ${#ADJECTIVES[@]}]}_${NOUNS[RANDOM % ${#NOUNS[@]}]}"
        run_name_taken "$name" || { echo "$name"; return 0; }
    done
    echo "${name}_$(date +%Y%m%d%H%M%S)"
}
LATEST_FILE="run_all/$SCOPE.latest"   # the scope's most recently started run
if [ -z "$RUN_NAME" ]; then
    if [ "$ACTION" = run ]; then
        RUN_NAME=$(new_run_name)
        ORIG_ARGS+=(--run-name "$RUN_NAME")   # so the detached driver gets the same name
    elif [ -s "$LATEST_FILE" ]; then
        RUN_NAME=$(cat "$LATEST_FILE")
    elif [ "$ACTION" = plan ]; then
        RUN_NAME="<run>"
    else
        echo "== no $SCOPE run started yet (give one with --run-name) ==" >&2
        exit 1
    fi
fi
if [ "$RUN_NAME" != "<run>" ] && ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "== --run-name: letters, digits, '_', '.' and '-' only (got '$RUN_NAME') ==" >&2
    exit 1
fi

# The run's configs: a copy of the base config whose work/cache/results
# directories are under <work_dir>/<run>, and the tune stage's tuned config
# next to it. Where compute nodes can't write the repo, tuned_configs is a
# link to $DT_RESULTS_DIR/tuned-configs (jobs/setup_bash.sh, or below).
RUN_CONFIG_DIR="tuned_configs/$RUN_NAME"
CONFIG="$RUN_CONFIG_DIR/$(basename "$BASE_CONFIG")"
TUNED="${CONFIG%.yaml}_tuned.yaml"
RERUN="bash jobs/run_all.sh $SCOPE --run-name $RUN_NAME"
[ "$UNATTENDED" -eq 1 ] && RERUN+=" --unattended"
CONTINUE="$RERUN"                   # after an --until stop: the same command without --until
[ -n "$UNTIL" ] && RERUN+=" --until $UNTIL"

RUN_DIR="run_all/$RUN_NAME"
STATE="$RUN_DIR/$SCOPE.state"       # lines: <stage> <status> <round> <job ids...>
HEARTBEAT="$RUN_DIR/$SCOPE.heartbeat"
DRIVER="$RUN_DIR/$SCOPE.driver"     # "<host> <pid>" of the live driver
STOP_FILE="$RUN_DIR/$SCOPE.stop"
# Every job this run submits writes its Slurm output and its Ollama log here
# (sbatch --output, and DT_LOG_DIR in the job's environment), next to the
# driver's own log.
RUN_LOG_DIR="$DT_LOGS_ROOT/$RUN_NAME"
LOG="$RUN_LOG_DIR/run_all-$SCOPE.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

# Minutes -> "7h 00m" / "45m".
human() { if [ "$1" -ge 60 ]; then printf '%dh %02dm' $(($1 / 60)) $(($1 % 60)); else printf '%dm' "$1"; fi; }
# Minutes -> Slurm --time (HH:MM:00).
hhmm() { printf '%02d:%02d:00' $(($1 / 60)) $(($1 % 60)); }

# A config's paths.<key>, resolved as the jobs see them (repo-relative -> absolute).
cfg_path() {
    "$DT_SYS_PYTHON" -c 'import sys; sys.path.insert(0, "src")
from common import load_config
print(load_config(sys.argv[1])["paths"][sys.argv[2]])' "$1" "$2"
}

# (Re)write the run's copy of the base config (common.write_run_config).
write_run_config() {
    if [ "${DT_WORKER_WRITES_REPO:-1}" != 1 ] && [ ! -e tuned_configs ]; then
        mkdir -p "$DT_RESULTS_DIR/tuned-configs" && ln -s "$DT_RESULTS_DIR/tuned-configs" tuned_configs || return 1
    fi
    "$DT_SYS_PYTHON" -c 'import sys; sys.path.insert(0, "src")
from common import write_run_config
write_run_config(*sys.argv[1:])' "$BASE_CONFIG" "$RUN_NAME" "$CONFIG"
}

# --- state file ---------------------------------------------------------------
state_line()   { [ -f "$STATE" ] && awk -v s="$1" '$1 == s' "$STATE" | tail -n 1; }
state_status() { state_line "$1" | awk '{print $2}'; }
state_round()  { state_line "$1" | awk '{print $3}'; }
state_jobs()   { state_line "$1" | awk '{for (i = 4; i <= NF; i++) printf "%s%s", $i, (i < NF ? " " : ""); print ""}'; }
state_set() {   # <stage> <status> <round> [job ids...]
    local tmp="$STATE.tmp.$$"
    { [ -f "$STATE" ] && awk -v s="$1" '$1 != s' "$STATE"; echo "$*"; } > "$tmp" && mv "$tmp" "$STATE"
}

# --- driver liveness (works from any login node: the files are shared) --------
driver_alive() {
    [ -f "$HEARTBEAT" ] || return 1
    [ $(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") )) -lt 300 ]
}

# --- planning ------------------------------------------------------------------
# Time limits (minutes), one per job, for <est> minutes of work.
plan_chain() {
    local est=$1 m
    while [ "$est" -gt "$WORK_PER_JOB" ]; do
        echo "$MAX_JOB_MIN"
        est=$((est - WORK_PER_JOB))
    done
    m=$(( (est + MARGIN_MIN + 14) / 15 * 15 ))      # rounded up to 15 min
    [ "$m" -gt "$MAX_JOB_MIN" ] && m=$MAX_JOB_MIN
    echo "$m"
}

stage_config() { case "$1" in models|resolve|extract|tune) echo "$CONFIG" ;; *) echo "$TUNED" ;; esac; }

show_plan() {
    local stage limits total=0 m after_until=0
    echo "== $SCOPE scope, run $RUN_NAME ($CONFIG): planned jobs per stage (est. work -> time limits) =="
    for stage in "${STAGES[@]}"; do
        if [ "$after_until" -eq 1 ]; then
            printf '   %-10s not run (--until %s)\n' "$stage" "$UNTIL"
            continue
        fi
        [ "$stage" = "$UNTIL" ] && after_until=1
        limits=""
        if on_login_node "$stage"; then
            limits="on the login node, no Slurm job (workers have no internet) "
        else
            for m in $(plan_chain "${EST[$stage]}"); do limits+="$(hhmm "$m") "; done
        fi
        total=$((total + EST[$stage]))
        printf '   %-10s %-8s -> %s(%s)\n' "$stage" "$(human "${EST[$stage]}")" "$limits" "$(stage_config "$stage")"
    done
    echo "   total estimated work: $(human "$total"), plus queue waits"
}

show_status() {
    local stage st
    echo "== $SCOPE scope, run $RUN_NAME: stage status ($STATE) =="
    for stage in "${STAGES[@]}"; do
        st=$(state_status "$stage")
        printf '   %-10s %-8s round %-2s jobs %s\n' "$stage" "${st:-not started}" "$(state_round "$stage")" "$(state_jobs "$stage")"
    done
    if driver_alive; then echo "== driver running: $(cat "$DRIVER" 2>/dev/null) (log: $LOG) =="
    else echo "== no driver running — re-run '$RERUN' to resume =="; fi
    echo "== logs: $RUN_LOG_DIR/ =="
    echo "== runs: $(find run_all -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null)=="
    return 0
}

# --- Slurm ---------------------------------------------------------------------
# Final state of a job (COMPLETED, TIMEOUT, FAILED, ...) or "" while it's
# still queued or running.
final_state() {
    local st
    st=$(sacct -j "$1" -X -n -P -o State 2>/dev/null | head -n 1 | awk '{print $1}')
    case "$st" in
        ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING|SIGNALING|STAGE_OUT|REQUEUE_*) echo "" ;;
        *) echo "$st" ;;
    esac
}

# Block until a job ends (monitor-job.sh logs its progress); its final
# state goes in WAIT_STATE. Waiting is done with `wait` on background
# children, which a signal interrupts — a foreground child or $(...) would
# hold off the driver's INT/TERM/HUP trap until the job ended.
WAIT_STATE=""
wait_job() {
    local tries=0
    WAIT_STATE=$(final_state "$1")
    while [ -z "$WAIT_STATE" ]; do
        bash jobs/monitor-job.sh --no-lock -i "$INTERVAL" "$1" >&2 &
        wait $!
        WAIT_STATE=$(final_state "$1")
        [ -n "$WAIT_STATE" ] && break
        # sacct can lag behind squeue for a moment after a job ends, and a
        # failed Slurm query makes the monitor return early: check again.
        tries=$((tries + 1))
        if [ "$tries" -gt 10 ]; then WAIT_STATE=UNKNOWN; break; fi
        sleep "$INTERVAL" &
        wait $!
        WAIT_STATE=$(final_state "$1")
    done
}

cancel_pending() {
    local id
    for id in "$@"; do
        [ -z "$(final_state "$id")" ] && scancel "$id" 2>/dev/null && log "   cancelled job $id (no longer needed)"
    done
    return 0
}

# Submit a chain of jobs for <est> minutes of work; print their ids.
submit_chain() {
    local stage=$1 est=$2 config ids=() dep m jid
    config=$(stage_config "$stage")
    for m in $(plan_chain "$est"); do
        dep=()
        if [ ${#ids[@]} -gt 0 ]; then
            dep=(--dependency="afternotok:$(IFS=:; echo "${ids[*]}")")
            # Trillium's sbatch rejects --kill-on-invalid-dep ("option not
            # recognized"). It is only a backstop there: watch_chain cancels
            # the rest of a chain once a job completes.
            [ "${DT_CLUSTER:-}" != trillium ] && dep+=(--kill-on-invalid-dep=yes)
        fi
        # --log-dir: the job's Ollama log goes to this run's folder too.
        if ! jid=$(bash jobs/submit.sh --parsable --time="$(hhmm "$m")" --output="$DT_LOG_DIR/%x-%j.out" "${dep[@]}" "${SCRIPT[$stage]}" "$config" "${PROFILE_ARGS[@]}" --log-dir "$DT_LOG_DIR"); then
            log "$stage: sbatch failed"
            [ ${#ids[@]} -gt 0 ] && scancel "${ids[@]}"
            return 1
        fi
        jid=${jid%%;*}
        log "   submitted job $jid: ${SCRIPT[$stage]} $config --time=$(hhmm "$m")${dep[0]:+ ${dep[0]}}"
        ids+=("$jid")
    done
    echo "${ids[*]}"
}

# Wait for a chain. Returns 0 if a job COMPLETED, 2 if the chain ended
# without that but may be continued, 1 to stop (retryable when unattended),
# 3 to stop for good (a job was cancelled).
FAILS=0
watch_chain() {
    local stage=$1; shift
    local ids=("$@") i st rest out
    for ((i = 0; i < ${#ids[@]}; i++)); do
        rest=("${ids[@]:i+1}")
        log "$stage: waiting for job ${ids[i]} ($((i + 1)) of ${#ids[@]} in this chain)"
        wait_job "${ids[i]}"
        st=$WAIT_STATE
        log "$stage: job ${ids[i]} ended: $st"
        case "$st" in
            COMPLETED)
                FAILS=0
                cancel_pending "${rest[@]}"
                return 0 ;;
            TIMEOUT)
                FAILS=0
                [ ${#rest[@]} -gt 0 ] && log "$stage: time limit reached — job ${rest[0]} continues from the checkpoints" ;;
            CANCELLED*)
                log "$stage: job ${ids[i]} was cancelled (by you or an administrator) — stopping"
                cancel_pending "${rest[@]}"
                return 3 ;;
            *)
                FAILS=$((FAILS + 1))
                out=$(ls "$DT_LOGS_ROOT"/*-"${ids[i]}".out "$DT_LOGS_ROOT"/*/*-"${ids[i]}".out 2>/dev/null | head -n 1)
                log "$stage: job ${ids[i]} $st — see ${out:-its .out file}"
                if [ "$FAILS" -ge "$MAX_FAILS" ]; then
                    log "$stage: $FAILS jobs in a row failed — stopping"
                    cancel_pending "${rest[@]}"
                    return 1
                fi
                [ ${#rest[@]} -gt 0 ] && log "$stage: job ${rest[0]} retries from the checkpoints" ;;
        esac
    done
    return 2
}

outputs_exist() {
    case "$1" in
        resolve) [ -f "$(cfg_path "$CONFIG" cache_dir)/item_mapping.json" ] ;;
        extract) local w; w=$(cfg_path "$CONFIG" work_dir)
                 [ -f "$w/cohort.parquet" ] && [ -f "$w/panel_long.parquet" ] ;;
        *) return 1 ;;
    esac
}

# Stages that download from the internet; on clusters whose worker nodes
# have none (environment.workers_have_internet: false), they run here on the
# login node instead of as Slurm jobs.
needs_internet() { [ "$1" = models ]; }
on_login_node() { needs_internet "$1" && [ "${DT_WORKER_INTERNET:-1}" != 1 ]; }

# Run a stage's job script directly on this login node (no Slurm), logging to
# this run's log directory. Its script resumes where it left off, so a run
# interrupted by a driver restart is simply started again.
run_on_login_node() {
    local stage=$1 out
    out="$DT_LOG_DIR/${stage}-login-node-$(date +%Y%m%d-%H%M%S).out"
    log "$stage: worker nodes have no internet access (environment.workers_have_internet: false)"
    log "$stage: running ${SCRIPT[$stage]} on this login node instead of submitting it; output: $out"
    state_set "$stage" running 1 login-node
    if bash "${SCRIPT[$stage]}" "$(stage_config "$stage")" "${PROFILE_ARGS[@]}" --log-dir "$DT_LOG_DIR" > "$out" 2>&1; then
        state_set "$stage" done 1 login-node
        log "$stage: COMPLETED"
        return 0
    fi
    state_set "$stage" failed 1 login-node
    log "$stage: FAILED on the login node — see $out"
    return 1
}

run_stage() {
    local stage=$1 status round jobs est rc max_rounds=$MAX_ROUNDS
    [ "$UNATTENDED" -eq 1 ] && max_rounds=$MAX_ROUNDS_UNATTENDED
    status=$(state_status "$stage")
    case "$status" in
        done|skipped) log "$stage: done earlier"; return 0 ;;
        review) log "$stage: item mapping reviewed — continuing"; state_set "$stage" done "$(state_round "$stage")" $(state_jobs "$stage"); return 0 ;;
    esac
    if on_login_node "$stage"; then
        run_on_login_node "$stage"
        return $?
    fi
    if [ -z "$status" ] && [ "$REDO_EXTRACT" -eq 0 ] && outputs_exist "$stage"; then
        log "$stage: outputs already exist — skipped (use --redo-extract to re-run)"
        state_set "$stage" skipped 0
        return 0
    fi
    jobs=$(state_jobs "$stage")
    [ "$jobs" = login-node ] && jobs=""      # was run on a login node; nothing to re-attach to
    round=$(state_round "$stage"); round=${round:-0}
    if [ "$status" = failed ]; then
        log "$stage: failed last time — resubmitting from its checkpoints"
        jobs=""; round=0
    fi
    [ -n "$jobs" ] && log "$stage: re-attaching to jobs $jobs"
    while true; do
        if [ -z "$jobs" ]; then
            round=$((round + 1))
            if [ "$round" -gt $((1 + max_rounds)) ]; then
                log "$stage: still unfinished after $max_rounds extra jobs — stopping; check the .out files and docs/runtime_estimates.md"
                state_set "$stage" failed "$((round - 1))"
                return 1
            fi
            if [ ! -f "$(stage_config "$stage")" ]; then
                log "$stage: $(stage_config "$stage") not found (written by the tune stage) — stopping"
                return 3
            fi
            est=${EST[$stage]}
            # A further round is one job: the stage's estimate, at most 7 h of work.
            [ "$round" -gt 1 ] && [ "$est" -gt "$WORK_PER_JOB" ] && est=$WORK_PER_JOB
            log "$stage: submitting (round $round, ~$(human "$est") of work)"
            jobs=$(submit_chain "$stage" "$est") || { state_set "$stage" failed "$round"; return 1; }
            state_set "$stage" running "$round" $jobs
        fi
        watch_chain "$stage" $jobs
        rc=$?
        case $rc in
            0) if [ "$stage" = resolve ] && [ "$UNATTENDED" -eq 0 ]; then state_set "$stage" review "$round" $jobs
               else state_set "$stage" done "$round" $jobs; fi
               [ "$stage" = resolve ] && [ "$UNATTENDED" -eq 1 ] && \
                   log "$stage: unattended — not pausing; review $(cfg_path "$CONFIG" cache_dir)/item_mapping.json afterwards"
               log "$stage: COMPLETED"
               return 0 ;;
            1|3) state_set "$stage" failed "$round" $jobs; return $rc ;;
            2) log "$stage: its jobs ended without finishing — submitting another"; jobs="" ;;
        esac
    done
}

# run_stage, plus --unattended's retries: a stage that stopped on failures
# is resubmitted from its checkpoints after a cooldown (15, 30, 60, 120,
# 120 min). A cancelled job or a missing config (return 3) is not retried.
run_stage_retrying() {
    local stage=$1 attempt=0 rc wait_min
    while true; do
        run_stage "$stage"
        rc=$?
        [ "$rc" -eq 0 ] && return 0
        if [ "$UNATTENDED" -eq 0 ] || [ "$rc" -ne 1 ] || [ "$attempt" -ge "$MAX_STAGE_RETRIES" ]; then
            [ "$UNATTENDED" -eq 1 ] && [ "$rc" -eq 1 ] && log "$stage: $MAX_STAGE_RETRIES retries used up — giving up"
            return 1
        fi
        attempt=$((attempt + 1))
        wait_min=$((COOLDOWN_MIN << (attempt - 1)))
        [ "$wait_min" -gt 120 ] && wait_min=120
        log "$stage: unattended — retry $attempt of $MAX_STAGE_RETRIES in $wait_min min, from its checkpoints"
        sleep $((wait_min * 60)) &
        wait $!
        FAILS=0
    done
}

# --- actions -------------------------------------------------------------------
case "$ACTION" in
    plan) show_plan; exit 0 ;;
    status) show_status; exit 0 ;;
    stop)
        if driver_alive; then
            touch "$STOP_FILE"
            echo "== asked the driver ($(cat "$DRIVER" 2>/dev/null)) to stop; it will within a minute. Submitted jobs keep running: =="
            squeue -u "$USER" -h -o '   %i %j %T %M/%l' 2>/dev/null
        else
            echo "== no driver running for $SCOPE, run $RUN_NAME =="
        fi
        exit 0 ;;
esac

mkdir -p "$RUN_DIR" "$RUN_LOG_DIR"
echo "$RUN_NAME" > "$LATEST_FILE"

# Detach: start the driver in its own session, then follow its log.
if [ "$FOREGROUND" -eq 0 ]; then
    echo "== run name: $RUN_NAME (resume with: $RERUN) =="
    if driver_alive; then
        echo "== a driver is already running ($(cat "$DRIVER" 2>/dev/null)) — following its log =="
        [ -n "$REDO" ] && echo "== --redo ignored: it applies when a driver starts; stop this one first (--stop) ==" >&2
        [ -n "$UNTIL" ] && echo "== --until ignored: it applies when a driver starts; stop this one first (--stop) ==" >&2
        read -r d_host d_pid 2>/dev/null < "$DRIVER"
    else
        rm -f "$STOP_FILE"
        RUN_ALL_DETACHED=1 setsid nohup bash "$0" "${ORIG_ARGS[@]}" --foreground >> "$LOG" 2>&1 < /dev/null &
        d_host=$(hostname); d_pid=$!
        echo "== driver started in the background (host $d_host, pid $d_pid); log: $LOG =="
    fi
    echo "== following the log — Ctrl+C or a dropped connection stops only the following;"
    echo "== run the same command again to follow it, --status to check, --stop to stop it =="
    if [ "${d_host:-}" = "$(hostname)" ] && [ -n "${d_pid:-}" ]; then
        exec tail -n 30 -F --pid="$d_pid" "$LOG"
    fi
    exec tail -n 30 -F "$LOG"
fi

# --- the driver ------------------------------------------------------------------
if driver_alive; then
    echo "== a driver for $SCOPE, run $RUN_NAME is already running ($(cat "$DRIVER" 2>/dev/null)) — not starting another ==" >&2
    exit 1
fi
TEE_PID=""
if [ -z "${RUN_ALL_DETACHED:-}" ]; then
    exec > >(tee -a "$LOG") 2>&1
    TEE_PID=$!                          # spared by kill_tree; ends when the driver does
fi
echo "$(hostname) $$" > "$DRIVER"
touch "$HEARTBEAT"

DT_LOG_DIR="$RUN_LOG_DIR"
export DT_LOG_DIR

# TERM a process and everything below it (except the caller), children first.
# Bash runs the driver's TERM trap only once its current child (a
# monitor-job.sh that may wait for hours) has ended, so those go too.
kill_tree() {
    local child
    for child in $(pgrep -P "$1"); do
        [ "$child" = "$BASHPID" ] || [ "$child" = "$TEE_PID" ] || kill_tree "$child"
    done
    [ "$1" = "$BASHPID" ] || kill -TERM "$1" 2>/dev/null
}

# Heartbeat (lets --status/--stop and a second copy see this driver from any
# login node) and stop requests. No Slurm queries here.
(
    while kill -0 $$ 2>/dev/null; do
        touch "$HEARTBEAT"
        if [ -e "$STOP_FILE" ]; then
            rm -f "$STOP_FILE"
            kill_tree $$
            exit 0
        fi
        sleep 30
    done
) &
HEARTBEAT_PID=$!

stop_driver() {
    trap - INT TERM HUP
    log "driver stopped — submitted jobs keep running; re-run '$RERUN' to resume"
    kill_tree $$                        # monitor-job.sh, sleep, heartbeat
    exit 130
}
trap stop_driver INT TERM HUP
trap 'kill "$HEARTBEAT_PID" 2>/dev/null; rm -f "$HEARTBEAT" "$DRIVER"' EXIT

# --redo: forget these stages' status so they run again (only once, by this
# driver; a later re-attach without --redo leaves them as they are).
if [ -n "$REDO" ]; then
    for stage in ${REDO//,/ }; do
        if [[ " ${STAGES[*]} " != *" $stage "* ]]; then
            log "== --redo: unknown stage '$stage' (stages: ${STAGES[*]}) =="; exit 1
        fi
        [ -f "$STATE" ] && awk -v s="$stage" '$1 != s' "$STATE" > "$STATE.tmp.$$" && mv "$STATE.tmp.$$" "$STATE"
        log "== --redo: stage '$stage' will run again =="
    done
fi
if ! write_run_config; then
    log "== could not write $CONFIG from $BASE_CONFIG =="; exit 1
fi
log "== run_all $SCOPE, run $RUN_NAME: $CONFIG, checking Slurm every ${INTERVAL}s (host $(hostname), pid $$) =="
log "== logs of this run: $RUN_LOG_DIR/; results: $(cfg_path "$CONFIG" work_dir)/ =="
show_plan >&2

for stage in "${STAGES[@]}"; do
    if ! run_stage_retrying "$stage"; then
        log "== stopped at stage '$stage'. Fix the cause, then re-run: $RERUN =="
        exit 1
    fi
    if [ "$(state_status "$stage")" = review ]; then
        log "== item mapping written: $(cfg_path "$CONFIG" cache_dir)/item_mapping.json"
        log "== REVIEW it (see docs/resolve_items.md), then run the same command again to continue =="
        exit 3
    fi
    if [ "$stage" = "$UNTIL" ]; then
        log "== --until $UNTIL: stopping after stage '$stage' as asked. To go on with the next stages: $CONTINUE =="
        exit 0
    fi
done

log "== all stages done. RQ1-RQ3: $(cfg_path "$TUNED" results_dir)/statistical_analysis.json =="
log "== per-condition metrics: $(cfg_path "$TUNED" results_dir)/all_conditions_summary.csv =="
# The run's logs, results and configs in one tar next to the work
# directories ($DT_RESULTS_DIR/<run>_results.tar), ready to copy off scratch.
log "== packing the run: jobs/pack_run.sh $RUN_NAME =="
if bash jobs/pack_run.sh "$RUN_NAME" -o "$DT_RESULTS_DIR" "${PROFILE_ARGS[@]}" >&2; then
    log "== packed: $DT_RESULTS_DIR/${RUN_NAME}_results.tar =="
else
    log "== packing failed — run it by hand: bash jobs/pack_run.sh $RUN_NAME =="
fi
