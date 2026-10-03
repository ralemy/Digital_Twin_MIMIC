#!/bin/bash
# =============================================================================
# Watch one Slurm job until it ends — run on a LOGIN NODE, not with sbatch:
#   bash jobs/monitor-job.sh                # your first job, as listed by `sq`
#   bash jobs/monitor-job.sh 23132444       # a specific job
#   bash jobs/monitor-job.sh -i 120 23132444   # check every 2 minutes
#
# Every minute it prints one status line (state, elapsed / time limit, node)
# and, while the job runs, the latest line of its output file whenever that
# line has changed (e.g. "batch 7/15 done").
# While the job is PENDING it also prints, on the first check, every 5
# minutes after that and whenever the pending reason changes, what helps
# judge when it may start:
#   - the pending reason in plain words, the scheduler's estimated start
#     time (if it has computed one) and how long the job has waited;
#   - what the job asked for (partitions, CPUs, memory, GPUs, time limit);
#   - how many pending jobs in the same partitions have a higher priority
#     (are ahead of it), and how many jobs are running there;
#   - for GPU jobs, the GPUs of the requested type in those partitions:
#     free now, free but held for higher-priority jobs, busy, and on
#     drained/down nodes;
#   - node and CPU counts per partition, the job's priority breakdown
#     (sprio) and your account's fair-share.
# When the job ends it prints its final state, exit code and run time from
# sacct (plus `seff` efficiency if available) and exits 0 if the job
# COMPLETED, 1 otherwise — so it can be chained:
#   bash jobs/monitor-job.sh 23132444 && echo "done, go look"
#
# Cluster etiquette (Alliance): it queries Slurm at most once per interval
# (minimum 60s), only one copy runs per user at a time, and it refuses to
# run inside a Slurm job. Stop it any time with Ctrl+C — that only stops
# the monitor, not the job. --no-lock skips the one-copy check; it's for
# jobs/run_all.sh, which holds its own lock and runs one monitor at a time.
# =============================================================================
set -uo pipefail

INTERVAL=60
DETAIL_EVERY=5          # pending details every this many checks
JOB_ID=""
USE_LOCK=1

usage() { sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--interval) INTERVAL="${2:-}"; shift 2 ;;
        --no-lock) USE_LOCK=0; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown option: $1" >&2; usage 1 ;;
        *) JOB_ID="$1"; shift ;;
    esac
done
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 60 ]; then
    echo "== interval must be a whole number of seconds, at least 60 (Alliance: don't poll Slurm more often) ==" >&2
    exit 1
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "== run this on a login node, not inside a Slurm job (SLURM_JOB_ID=$SLURM_JOB_ID) ==" >&2
    exit 1
fi

# One monitor per user at a time.
if [ "$USE_LOCK" -eq 1 ]; then
    LOCK="${XDG_RUNTIME_DIR:-/tmp}/monitor-job.$USER.lock"
    exec 9> "$LOCK"
    if ! flock -n 9; then
        echo "== another monitor-job.sh is already running for $USER — stop it first (only one monitoring loop at a time) ==" >&2
        exit 1
    fi
fi

if [ -z "$JOB_ID" ]; then
    JOB_ID=$(squeue -u "$USER" -h -o %i | head -1)
    if [ -z "$JOB_ID" ]; then
        echo "== you have no jobs in the queue (sq is empty) ==" >&2
        exit 1
    fi
    echo "== no job id given — monitoring your first job in sq: $JOB_ID =="
fi

trap 'echo; echo "== monitor stopped (the job itself is unaffected) =="; exit 130' INT TERM

ts() { date '+%H:%M:%S'; }

# Seconds → "1h 05m" / "12m" / "40s".
human() {
    local s=$1
    if [ "$s" -ge 3600 ]; then printf '%dh %02dm' $((s / 3600)) $((s % 3600 / 60))
    elif [ "$s" -ge 60 ]; then printf '%dm' $((s / 60))
    else printf '%ds' "$s"; fi
}

explain_reason() {
    case "$1" in
        Priority) echo "other pending jobs that want the same resources have higher priority; it starts when it reaches the front" ;;
        Resources) echo "it is at the front of the line and is waiting for enough resources (nodes/GPUs/memory) to free up" ;;
        ReqNodeNotAvail*|*"reserved for other job"*) echo "the nodes it could use are busy, drained, or held for higher-priority jobs (often a large job being backfilled around)" ;;
        Dependency) echo "waiting for another job it depends on (--dependency) to finish" ;;
        DependencyNeverSatisfied) echo "a job it depends on ended in a way that will never satisfy the dependency — cancel it" ;;
        BeginTime) echo "it was submitted with --begin and that time hasn't come yet" ;;
        JobHeldUser) echo "held by you (scontrol hold) — release with: scontrol release $JOB_ID" ;;
        JobHeldAdmin) echo "held by an administrator — contact support" ;;
        QOS*Limit*|QOSMax*|AssocGrp*|AssocMax*) echo "you or your group are at a usage limit (e.g. GPUs or jobs at once); it starts when your other jobs finish" ;;
        PartitionTimeLimit) echo "its --time is longer than these partitions allow — it will never start; cancel and resubmit with a shorter --time" ;;
        PartitionNodeLimit|BadConstraints) echo "it asks for something no node in these partitions can provide — check the request" ;;
        None) echo "the scheduler hasn't evaluated it yet (just submitted)" ;;
        *) echo "see 'Job Reason Codes' in the Slurm squeue docs" ;;
    esac
}

# GPUs of a type across nodes in the given partitions, from sinfo per node.
# Prints: total busy free_now free_planned unavailable nodes_with_free max_free_on_one_node
gpu_summary() {
    local parts=$1 type=$2
    sinfo -p "$parts" -N -h -O "NodeHost:60,StateCompact:20,Gres:2000,GresUsed:2000" 2>/dev/null | awk -v t="$type" '
        function gpus(s,   tok, a, sum) {
            sum = 0
            while (match(s, /gpu:[^:,()]+:[0-9]+/)) {
                tok = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH)
                split(tok, a, ":")
                if (t == "" || a[2] == t) sum += a[3]
            }
            return sum
        }
        !seen[$1]++ {
            state = $2; tot = gpus($3); used = gpus($4); free = tot - used
            if (tot == 0) next
            T += tot; U += used
            if (state ~ /^(drain|drng|down|fail|maint|inval|resv|futr)/) { X += tot; next }
            if (free <= 0) next
            if (state ~ /-$/) { P += free; next }    # "planned": held for a higher-priority job
            F += free; NFREE++; if (free > M) M = free
        }
        END { printf "%d %d %d %d %d %d %d\n", T, U, F, P, X, NFREE, M }'
}

show_pending_details() {
    local reason=$1 parts=$2 gres=$3 cpus=$4 mem=$5 limit=$6 submit=$7 start=$8 prio=$9 account=${10}
    local now waited
    now=$(date +%s)
    waited=$(( now - $(date -d "$submit" +%s 2>/dev/null || echo "$now") ))

    echo "   ┌─ pending: $reason"
    echo "   │  meaning: $(explain_reason "$reason")"
    echo "   │  waited: $(human "$waited") (submitted $submit)"
    if [[ "$start" =~ ^[0-9]{4}- ]]; then
        echo "   │  scheduler's estimated start: $start  (an estimate; it moves as jobs finish early or new ones arrive)"
    else
        echo "   │  scheduler's estimated start: not computed yet"
    fi
    echo "   │  requested: partitions=$parts  cpus=$cpus  mem=$mem  gpus=${gres#gres/}  time=$limit"

    # Jobs ahead in the same partitions.
    local queue n_pending n_ahead n_running
    queue=$(squeue -p "$parts" -t PD,R -h -o "%i|%T|%Q" 2>/dev/null | sort -u)
    n_pending=$(awk -F'|' '$2 == "PENDING"' <<< "$queue" | wc -l)
    n_ahead=$(awk -F'|' -v p="$prio" -v me="$JOB_ID" '$2 == "PENDING" && $1 != me && $3 > p' <<< "$queue" | wc -l)
    n_running=$(awk -F'|' '$2 == "RUNNING"' <<< "$queue" | wc -l)
    echo "   │  queue in these partitions: $n_ahead pending jobs ahead of yours (higher priority), $n_pending pending in total, $n_running running"

    # GPUs of the requested type.
    if [[ "$gres" == *gpu* ]]; then
        local spec type want T U F P X NF M
        spec=${gres#*gpu:}                      # e.g. "h100:1" or "1"
        if [[ "$spec" == *:* ]]; then type=${spec%%:*}; want=${spec##*:}; else type=""; want=$spec; fi
        read -r T U F P X NF M < <(gpu_summary "$parts" "$type")
        echo "   │  ${type:-any}-type GPUs in these partitions: $T total, $U busy, $F free now on $NF nodes (max $M on one node),"
        echo "   │      $P free but held for higher-priority jobs, $X on drained/down nodes. You need $want per node."
        if [ "${F:-0}" -ge "${want:-1}" ] && [ "${M:-0}" -ge "${want:-1}" ]; then
            echo "   │      → enough GPUs are free right now; it's waiting on priority, CPUs/memory on those nodes, or a reservation."
        fi
    fi

    # Nodes and CPUs per partition (sinfo %F = nodes alloc/idle/other/total, %C = CPUs alloc/idle/other/total).
    echo "   │  partitions (nodes alloc/idle/other/total — CPUs alloc/idle/other/total):"
    sinfo -p "$parts" -h -o "%P|%F|%C" 2>/dev/null | awk -F'|' '{printf "   │      %-22s nodes %-16s CPUs %s\n", $1, $2, $3}'

    # Priority breakdown and fair-share (fair-share recovers as past usage decays).
    local sp
    sp=$(sprio -j "$JOB_ID" -h -o "%Y|%A|%F|%J|%P|%Q" 2>/dev/null | head -1)
    if [ -n "$sp" ]; then
        IFS='|' read -r p_tot p_age p_fs p_size p_part p_qos <<< "$sp"
        echo "   │  priority $p_tot = age $p_age + fair-share $p_fs + size $p_size + partition $p_part + qos $p_qos (age grows the longer it waits)"
    fi
    local fs
    fs=$(sshare -h -U -u "$USER" -A "$account" -o FairShare 2>/dev/null | awk 'NF {print $1; exit}')
    [ -n "$fs" ] && echo "   │  fair-share of account $account: $fs (0–1; higher = more priority; lower after heavy recent use)"
    echo "   └─"
}

show_final() {
    echo
    echo "== job $JOB_ID has left the queue — final record from sacct =="
    local rec state
    rec=$(sacct -j "$JOB_ID" -X -n -P -o JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList 2>/dev/null | head -1)
    if [ -z "$rec" ]; then
        echo "   (sacct has no record yet — check later with: sacct -j $JOB_ID)"
        return 1
    fi
    IFS='|' read -r _ name state code elapsed start end nodes <<< "$rec"
    echo "   name=$name  state=$state  exit=$code  elapsed=$elapsed  node=$nodes"
    echo "   started=$start  ended=$end"
    [ -n "$OUT_FILE" ] && echo "   output: $OUT_FILE"
    if command -v seff > /dev/null; then
        echo "   -- seff --"
        seff "$JOB_ID" 2>/dev/null | sed -n '/^State/,$p' | sed 's/^/   /'
    fi
    [[ "$state" == COMPLETED* ]]
}

# The output file's latest line, skipping per-request noise, cut to 200
# characters ("" if there's nothing yet).
latest_output() {
    [ -n "$OUT_FILE" ] && [ -r "$OUT_FILE" ] || return 0
    tail -n 200 "$OUT_FILE" 2>/dev/null | grep -v -e 'HTTP Request' -e '^\s*$' -e '^Traceback' -e '^  ' | tail -n 1 | cut -c1-200
}

OUT_FILE=""
LAST_REASON=""
LAST_OUTPUT=""
CHECK=0
SEEN=0
while true; do
    line=$(squeue -j "$JOB_ID" -h -o "%T|%r|%P|%b|%C|%m|%l|%M|%V|%S|%Q|%N|%j|%a" 2>/dev/null | head -1)
    if [ -z "$line" ]; then
        if [ "$SEEN" -eq 0 ] && [ -z "$(sacct -j "$JOB_ID" -X -n -o JobID 2>/dev/null)" ]; then
            echo "== job $JOB_ID not found in the queue or in sacct ==" >&2
            exit 1
        fi
        show_final; exit $?
    fi
    SEEN=1
    IFS='|' read -r state reason parts gres cpus mem limit used submit start prio nodes name account <<< "$line"

    if [ "$CHECK" -eq 0 ]; then
        echo "== monitoring job $JOB_ID ($name), checking every $(human "$INTERVAL") — Ctrl+C stops the monitor only =="
    fi
    if [ -z "$OUT_FILE" ]; then
        OUT_FILE=$(scontrol show job "$JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p')
        [ -n "$OUT_FILE" ] && [ "$CHECK" -eq 0 ] && echo "   output file: $OUT_FILE"
    fi

    case "$state" in
        PENDING)
            echo "[$(ts)] PENDING  ($reason)  time limit $limit"
            if [ $((CHECK % DETAIL_EVERY)) -eq 0 ] || [ "$reason" != "$LAST_REASON" ]; then
                show_pending_details "$reason" "$parts" "$gres" "$cpus" "$mem" "$limit" "$submit" "$start" "$prio" "$account"
            fi
            LAST_REASON=$reason ;;
        RUNNING|COMPLETING|CONFIGURING)
            [ "$LAST_REASON" != "__running__" ] && echo "[$(ts)] ▶ started on $nodes at $start"
            echo "[$(ts)] $state  elapsed $used / $limit  on $nodes"
            out=$(latest_output)
            if [ -n "$out" ] && [ "$out" != "$LAST_OUTPUT" ]; then
                echo "           │ $out"
                LAST_OUTPUT=$out
            fi
            LAST_REASON="__running__" ;;
        *)
            echo "[$(ts)] $state  ($reason)" ;;
    esac

    CHECK=$((CHECK + 1))
    sleep "$INTERVAL"
done
