#!/bin/bash
# =============================================================================
# Download progress that is readable in a log file — SOURCED, not submitted.
#
#   some_download 2>&1 | dt_progress "<label>"
#
# Tools like wget and `ollama pull` draw a live progress bar by rewriting one
# line with carriage returns and terminal codes: fine in a terminal, but in a
# Slurm .out file or a run_all log it becomes thousands of garbled updates.
# dt_progress turns that stream into plain lines:
#   - a progress line ("<label>: 45% ...") when the percentage has moved by
#     DT_PROGRESS_STEP points (default 10) or DT_PROGRESS_SECONDS (default 60)
#     have passed since the last one, plus the final 100%;
#   - every other non-empty line once (status messages such as
#     "verifying sha256 digest"), without consecutive repeats, except wget's
#     per-file connection lines (Resolving..., Connecting..., Saving to...).
# When standard output is a terminal (a download run by hand on a login node),
# it passes the stream through untouched, so the live bar shows as usual.
# =============================================================================

dt_progress() {
    local label=${1:-download}
    if [ -t 1 ]; then
        cat
        return
    fi
    LC_ALL=C awk -v label="$label" \
        -v step="${DT_PROGRESS_STEP:-10}" -v every="${DT_PROGRESS_SECONDS:-60}" '
        BEGIN { RS = "[\r\n]"; last_pct = -1; last_t = 0; prev = "" }
        {
            line = $0
            gsub(/\033\[[0-9;?]*[A-Za-z]/, "", line)   # terminal control codes
            gsub(/[\200-\377]+/, " ", line)            # bar glyphs (UTF-8 blocks)
            gsub(/\[[=> -]*\]/, " ", line)             # wget-style [====>   ] bar
            gsub(/[ \t]+/, " ", line); sub(/^ /, "", line); sub(/ $/, "", line)
            if (line == "") next
            # wget per-file connection chatter (the Length: and final saved lines stay)
            if (line ~ /^(--[0-9][0-9][0-9][0-9]-[0-9]|Resolving |Connecting to |Reusing existing connection|HTTP request sent|Saving to:)/) next
            if (match(line, /[0-9]+(\.[0-9]+)?%/)) {
                pct = substr(line, RSTART, RLENGTH - 1) + 0
                # wget starts its bar with the (possibly truncated) file name: drop it, the label says it
                split(line, tok, " ")
                if (tok[1] != "" && (index(label, tok[1]) == 1 || index(tok[1], label) == 1)) sub(/^[^ ]+ /, "", line)
                if (pct < last_pct) last_pct = -1       # a new item (e.g. the next model layer) started
                now = systime()
                if (pct >= last_pct + step || (pct > last_pct && now - last_t >= every) || (pct >= 100 && last_pct < 100)) {
                    printf "   %s: %s\n", label, line
                    fflush()
                    last_pct = pct; last_t = now
                }
                next
            }
            if (line != prev) { printf "   %s\n", line; fflush(); prev = line }
        }'
}
