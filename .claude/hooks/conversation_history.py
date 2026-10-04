#!/usr/bin/env python3
"""
Rebuild ~/thesis_methodology/conversation_history.md (outside the repo; the
repo's .gitignore also lists the old docs/ location) from every Claude Code
transcript of this project: a settings section, then each prompt followed by Claude's
final answer to it, oldest first.

Run by a Stop hook (.claude/settings.local.json) after every answer, so the
file always ends with the latest question and answer. Rebuilding from the
transcripts each time (rather than appending) keeps it correct after
interrupted turns, resumed sessions or a missed hook. Can also be run by
hand:  /usr/bin/python3 .claude/hooks/conversation_history.py

Reads the hook's JSON on stdin when there is one (its transcript_path gives
the transcript directory); otherwise uses this project's default directory.
Standard library only, so it runs with the system python3.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
OUT = Path.home() / "thesis_methodology" / "conversation_history.md"
DEFAULT_TRANSCRIPTS = Path.home() / ".claude" / "projects" / ("-" + str(PROJECT).strip("/").replace("/", "-").replace("_", "-"))

# Context the IDE or harness adds to a prompt; not what the user typed.
DROP_TAGS = ("ide_opened_file", "ide_selection", "system-reminder", "command-message",
             "local-command-stdout", "local-command-caveat", "task-notification")
SECRET = re.compile(r"(?i)\b(password|passwd|token|api[_-]?key|secret)\b(\s*[:=]\s*)(\S+)")


def clean_prompt(text: str) -> str:
    for tag in DROP_TAGS:
        text = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", text, flags=re.S)
    # Keep pasted text, as a quote block.
    text = re.sub(r"<pasted_content\b[^>]*>\s*(.*?)\s*</pasted_content>",
                  lambda m: "\n".join("> " + ln for ln in m.group(1).splitlines()), text, flags=re.S)
    text = re.sub(r"<command-name>(.*?)</command-name>", r"\1", text, flags=re.S)
    text = re.sub(r"<command-args>(.*?)</command-args>", r" \1", text, flags=re.S)
    return SECRET.sub(r"\1\2[redacted]", text).strip()


def texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]


def parse(path: str) -> dict:
    """One session: its metadata and its list of {prompt, followups, answer, time}."""
    turns, meta = [], {"models": set(), "versions": set(), "modes": set(), "cwd": None, "branch": None}
    current = None
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("isSidechain"):
            continue
        meta["versions"].add(r.get("version")) if r.get("version") else None
        meta["modes"].add(r.get("permissionMode")) if r.get("permissionMode") else None
        meta["cwd"] = meta["cwd"] or r.get("cwd")
        meta["branch"] = meta["branch"] or r.get("gitBranch")
        kind = r.get("type")
        if kind == "user" and not r.get("isMeta"):
            prompt = clean_prompt("\n\n".join(texts(r.get("message", {}).get("content"))))
            if prompt:
                current = {"prompt": prompt, "followups": [], "answer": "", "time": r.get("timestamp")}
                turns.append(current)
        elif kind == "attachment" and (r.get("attachment") or {}).get("type") == "queued_command":
            # A message sent while Claude was still answering: answered in the same response.
            text = clean_prompt("\n\n".join(texts(r["attachment"].get("prompt"))))
            if text and current is not None:
                current["followups"].append(text)
        elif kind == "assistant" and current is not None:
            msg = r.get("message", {})
            if msg.get("model") and not msg["model"].startswith("<"):
                meta["models"].add(msg["model"])
            body = "\n\n".join(t for t in texts(msg.get("content")) if t.strip())
            if body:
                current["answer"] = body      # the turn's last text is its final answer
    return {"path": path, "turns": turns, "meta": meta}


def settings_section(sessions: list[dict]) -> str:
    models = sorted(set().union(*(s["meta"]["models"] for s in sessions)))
    versions = sorted(set().union(*(s["meta"]["versions"] for s in sessions)))
    modes = sorted(set().union(*(s["meta"]["modes"] for s in sessions)))
    cwd = next((s["meta"]["cwd"] for s in sessions if s["meta"]["cwd"]), str(PROJECT))
    branch = next((s["meta"]["branch"] for s in sessions if s["meta"]["branch"]), "")
    instructions = [p for p in ("/etc/claude-code/CLAUDE.md", str(PROJECT / "CLAUDE.md"),
                                str(Path.home() / ".claude" / "CLAUDE.md")) if os.path.exists(p)]
    memory = DEFAULT_TRANSCRIPTS / "memory"
    lines = [
        "## Settings",
        "",
        "The configuration Claude worked under for the conversations below.",
        "",
        f"- **Assistant:** Claude Code (VS Code extension), model {', '.join(f'`{m}`' for m in models) or 'unknown'}",
        f"- **Claude Code versions:** {', '.join(versions) or 'unknown'}",
        f"- **Permission mode:** {', '.join(modes) or 'default'}",
        f"- **Working directory:** `{cwd}`" + (f" (git branch `{branch}`)" if branch else ""),
        "- **Machine:** Nibi login node (Digital Research Alliance of Canada); "
        "heavy work goes to Slurm jobs, not the login node",
        "- **Standing instructions:** " + (", ".join(f"`{p}`" for p in instructions)
                                           if instructions else "none found"),
    ]
    if instructions and instructions[0] == "/etc/claude-code/CLAUDE.md":
        lines += [
            "  - The organization policy (`/etc/claude-code/CLAUDE.md`) sets the Alliance cluster rules:",
            "    - keep login-node work light, and use `sbatch`/`salloc` for anything heavier;",
            "    - poll Slurm no more often than every 60 s, and never put sleeps in jobs;",
            "    - install Python packages from the Alliance wheelhouse into virtual environments;",
            "    - respect home/project/scratch quotas and purge rules;",
            "    - follow the Alliance documentation.",
        ]
    if memory.is_dir():
        lines.append(f"- **Claude's project memory:** `{memory}`")
    lines += [
        "- **Tools available:** file read/edit/write, shell, web search and fetch, "
        "sub-agents, skills (e.g. docx, update-config) and artifacts",
        f"- **Sessions included:** {len(sessions)} "
        f"(transcripts in `{Path(sessions[0]['path']).parent if sessions else DEFAULT_TRANSCRIPTS}`)",
        "",
        "Each entry below is a prompt as typed, minus IDE context and with secrets redacted. "
        "It is followed by Claude's final answer to it. Interim progress notes and tool calls are "
        "left out; the full detail is in the transcripts.",
        "",
    ]
    return "\n".join(lines)


def fmt_time(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts


def main() -> None:
    transcript_dir = DEFAULT_TRANSCRIPTS
    if not sys.stdin.isatty():
        try:
            hook = json.load(sys.stdin)
            if hook.get("transcript_path"):
                transcript_dir = Path(hook["transcript_path"]).parent
        except (json.JSONDecodeError, ValueError):
            pass
    files = sorted(glob.glob(str(transcript_dir / "*.jsonl")))
    sessions = [s for s in (parse(f) for f in files) if s["turns"]]
    sessions.sort(key=lambda s: s["turns"][0]["time"] or "")

    out = ["# Conversation history: MIMIC-IV digital twin (exp1)", "",
           f"_Regenerated {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')} "
           "by `.claude/hooks/conversation_history.py` (Stop hook)._", "",
           settings_section(sessions)]
    n = 0
    for s in sessions:
        if not any(t["answer"] or t["followups"] or not re.fullmatch(r"/\S+", t["prompt"]) for t in s["turns"]):
            continue
        out += [f"## Session {Path(s['path']).stem[:8]}: started {fmt_time(s['turns'][0]['time'])}", ""]
        for t in s["turns"]:
            if re.fullmatch(r"/\S+", t["prompt"]) and not t["answer"] and not t["followups"]:
                continue                      # a bare /exit, /clear ... with no answer
            n += 1
            out += [f"### Q{n} ({fmt_time(t['time'])})", "", t["prompt"], ""]
            for f in t["followups"]:
                out += ["_Follow-up sent while Claude was working:_", "", f, ""]
            out += ["**Answer:**", "", t["answer"] or "_(no text answer: interrupted or tool-only)_", "", "---", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".md.tmp")
    tmp.write_text("\n".join(out), encoding="utf-8")
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
