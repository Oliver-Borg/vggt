"""Render a Claude Code session transcript (.jsonl) as markdown: user messages and assistant text replies only
(no tool calls, tool results, thinking or system reminders).

Usage: python export_chat.py [session.jsonl] [out.md]
Default session: the newest transcript for this repo under ~/.claude/projects/."""
import glob, json, os, re, sys

def newest():
    d = os.path.expanduser("~/.claude/projects/" + os.getcwd().replace("/", "-"))
    return max(glob.glob(d + "/*.jsonl"), key=os.path.getmtime)

def text_of(content, role):
    if isinstance(content, str):
        return content
    return "\n\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")

def clean(t):
    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", t, flags=re.S)
    return t.strip()

def render(path):
    out, seen = [f"# Chat log: {os.path.basename(path)}\n"], set()
    for line in open(path):
        d = json.loads(line)
        t = d.get("type")
        if t not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
            continue
        m = d.get("message", {})
        txt = clean(text_of(m.get("content", ""), t))
        if not txt or txt.startswith("<command-") or txt.startswith("<local-command") or txt.startswith("[Request interrupted"):
            continue
        if t == "user" and txt.startswith("This session is being continued"):
            continue
        key = (t, d.get("uuid"))
        if key in seen:
            continue
        seen.add(key)
        ts = (d.get("timestamp") or "")[:19].replace("T", " ")
        out.append(f"## {'User' if t == 'user' else 'Assistant'} ({ts} UTC)\n\n{txt}\n")
    return "\n".join(out)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else newest()
    dst = sys.argv[2] if len(sys.argv) > 2 else "chat.md"
    open(dst, "w").write(render(src))
    print(dst, os.path.getsize(dst), "bytes")
