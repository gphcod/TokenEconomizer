"""Codex: the `exec --json` stream, and the rollout file an interactive session leaves.

`codex exec --json` emits one `item.completed` record per command or file change; those become
`shell` and `apply_patch` calls. The rollout (`~/.codex/sessions/.../rollout-*.jsonl`) wraps the
same item fields in `event_msg/item_completed` records with PascalCase item types, an argv list
for the command and a path-keyed map for changes; `codex_rollout_to_exec` rewrites it into exec
lines, after which every reader handles it as a harness stream. Usage comes from the rollout's
`token_count` events, converted as `te_harness.codex.row_usage` converts the exec stream's,
because Codex's `input_tokens` is the GRAND input total including both cache fields.

A compressed rollout (`.jsonl.zst`, as Codex keeps its older ones) is read where the standard
library can decompress it (`compression.zstd`, Python 3.14 on); elsewhere, or where it is corrupt,
it is reported as unreadable rather than half-read.

The rollouts Codex keeps are also a repository's history from before the plugin
(`plugin/history.py`): `codex_history_logs` lists them, `codex_rollout_head` says whose they are,
and `codex_prompt_turns` / `codex_turn_calls` replay each prompt-turn's calls as the live hooks
would have seen them.
"""
import json
import os
import re
import shlex

from .claude import HEAD_BYTES, LINE_BYTES, _epoch
from .jsonl import Unreadable, head_records, readable
from .jsonl import lines as _lines
from .jsonl import records as _records

_SHELL_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File:\s*(.+?)\s*$", re.M)
_SHELL_GIT_APPLY_PATH_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.M)


def _shell_patch_paths(command):
    """Paths touched by a patch applied through the shell, () when it is not one."""
    command = command or ""
    if "*** Begin Patch" in command:
        found = _SHELL_PATCH_FILE_RE.findall(command)
    elif re.search(r"\bgit\s+apply\b", command):
        found = _SHELL_GIT_APPLY_PATH_RE.findall(command)
    else:
        return ()
    return tuple(dict.fromkeys(p.strip() for p in found if p.strip()))


def codex_exec_calls(lines):
    """Normalize Codex `exec --json` item events.

    Commands become `shell` calls with a `command` argument; file-change items become
    `apply_patch` calls carrying the changed paths. Agent messages are not tool calls."""
    out = []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "item.completed":
            continue
        item = ev.get("item") or {}
        kind = item.get("type")
        if kind == "command_execution":
            command = item.get("command") or ""
            output = item.get("aggregated_output") or ""
            try:
                ok = int(item.get("exit_code")) == 0
            except (TypeError, ValueError):
                ok = False
            # A patch applied THROUGH THE SHELL (`apply_patch <<'PATCH' ... *** Begin
            # Patch`) is an edit as much as a native file_change item is; with some model
            # providers it is the only form the CLI emits. It is surfaced as the same
            # `apply_patch` call every reader already handles, so a provider change cannot
            # hide edits from them.
            paths = _shell_patch_paths(command)
            if paths and (ok or "Success." in output):
                out.append((
                    "apply_patch",
                    {"path": paths[0], "paths": list(paths), "changes": [],
                     "via": "shell"},
                    "",
                    True,
                ))
            out.append(("shell", {"command": command}, output, ok))
        elif kind == "file_change":
            changes = item.get("changes") or []
            paths = [c.get("path") for c in changes
                     if isinstance(c, dict) and c.get("path")]
            out.append((
                "apply_patch",
                {"path": paths[0] if paths else "", "paths": paths,
                 "changes": changes},
                "",
                item.get("status") == "completed",
            ))
    return out


_ROLLOUT_ITEM_TYPES = {
    "CommandExecution": "command_execution",
    "FileChange": "file_change",
    "AgentMessage": "agent_message",
    "ImageView": "view_image",
    "Extension": "web_search",
    "McpToolCall": "mcp_tool_call",
    "CollabToolCall": "collab_tool_call",
}


def _command_text(command):
    if isinstance(command, list):
        return shlex.join(str(c) for c in command)
    return str(command or "")


def rollout_item_to_exec(item):
    """One rollout item -> the `codex exec --json` item shape, or None to skip it."""
    kind = _ROLLOUT_ITEM_TYPES.get(item.get("type"), None)
    if kind is None:
        return None
    out = {"type": kind, "id": item.get("id")}
    if kind == "command_execution":
        out.update(command=_command_text(item.get("command")),
                   aggregated_output=item.get("aggregated_output")
                   or ((item.get("stdout") or "") + (item.get("stderr") or "")),
                   exit_code=item.get("exit_code"),
                   status=str(item.get("status") or "").lower() or "completed")
    elif kind == "file_change":
        changes = item.get("changes")
        if isinstance(changes, dict):
            changes = [dict({"path": p}, **({"kind": (v or {}).get("type")}
                                            if isinstance(v, dict) else {}))
                       for p, v in changes.items()]
        out.update(changes=changes or [],
                   status=str(item.get("status") or "").lower() or "completed")
    elif kind == "agent_message":
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        out.update(text=content or "")
    else:
        out.update({k: v for k, v in item.items() if k not in ("type", "id")})
    return out


def codex_rollout_to_exec(rollout_path, out_path):
    """Rewrite a Codex rollout into `codex exec --json` lines at `out_path`.

    Returns the number of items written, or None when the rollout cannot be read to its end
    (missing, corrupt, or compressed where this Python cannot decompress it). The thread id
    rides on a leading `thread.started` line, as in the exec stream."""
    if not rollout_path or not readable(rollout_path) or not os.path.exists(rollout_path):
        return None
    n = 0
    thread = None
    try:
        with open(out_path, "w", encoding="utf-8") as out:
            for line in _lines(rollout_path, strict=True):
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                if ev.get("type") == "session_meta" and thread is None:
                    thread = payload.get("id") or payload.get("session_id")
                    if thread:
                        out.write(json.dumps({"type": "thread.started",
                                              "thread_id": thread}) + "\n")
                    continue
                if ev.get("type") != "event_msg" or payload.get("type") != "item_completed":
                    continue
                item = rollout_item_to_exec(payload.get("item") or {})
                if item is None:
                    continue
                out.write(json.dumps({"type": "item.completed", "item": item}) + "\n")
                n += 1
    except Unreadable:
        return None
    return n


def codex_capability_trace(rollout_path, out_path):
    """The exec-stream lines `toolprune.codex.inspect_usage` reads, for a rollout.

    `codex_rollout_to_exec` keeps the items the other mechanisms read; capability evidence
    also needs the model's own tool calls. In code mode every call is one `exec` custom tool
    whose input calls `tools.<name>(...)` -- `tools.view_image(`, `tools.spawn_agent(` -- and
    those ride in `response_item` records, which are added here as `custom_tool_call` /
    `function_call` items, the shape `toolprune.codex._item_capabilities` parses. Returns the number
    of items written, or None when the rollout cannot be read."""
    n = codex_rollout_to_exec(rollout_path, out_path)
    if n is None:
        return None
    with open(out_path, "a", encoding="utf-8") as out:
        for line in _lines(rollout_path):
            if '"response_item"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            if ev.get("type") != "response_item" or p.get("type") not in (
                    "custom_tool_call", "function_call"):
                continue
            item = {"type": p.get("type"), "name": p.get("name"),
                    "id": p.get("call_id") or p.get("id"),
                    "status": str(p.get("status") or "completed").lower()}
            item["input" if p.get("type") == "custom_tool_call" else "arguments"] = (
                p.get("input") if p.get("type") == "custom_tool_call" else p.get("arguments"))
            out.write(json.dumps({"type": "item.completed", "item": item}) + "\n")
            n += 1
    return n


def codex_prompt_count(rollout_path):
    """How many prompts a Codex session ran: one turn each (`_turn_starts`)."""
    starts = _turn_starts(rollout_path)
    return sum(1 for d in _records(rollout_path) if starts(d))


def _turn_starts(rollout_path):
    """What opens a turn in this rollout: its `task_started` events (0.163.0), or where it has
    none, its `turn_context` records."""
    def started(d):
        return (d.get("type") == "event_msg"
                and (d.get("payload") or {}).get("type") == "task_started")
    if any(started(d) for d in _records(rollout_path)):
        return started
    return lambda d: d.get("type") == "turn_context"


# ---- the rollouts Codex keeps -----------------------------------------------------------------
# `rollout-<YYYY-MM-DDTHH-MM-SS>-<thread id>.jsonl`, compressed later to `.jsonl.zst`.
_ROLLOUT_NAME_RE = re.compile(
    r"rollout-(\d{4})-(\d{2})-(\d{2})T\d{2}-\d{2}-\d{2}-(.+?)\.jsonl(?:\.zst)?")
_SHELL_SCRIPT_FLAGS = {"-c", "-lc", "-command", "/c"}
HEAD_LINES = 20


def codex_home(env=None):
    """Where Codex keeps its state: `$CODEX_HOME`, else `~/.codex`."""
    env = os.environ if env is None else env
    if env.get("CODEX_HOME"):
        return env["CODEX_HOME"]
    home = env.get("HOME") if os.name != "nt" else env.get("USERPROFILE")
    return os.path.join(home or os.path.expanduser("~"), ".codex")


def codex_rollout_id(path):
    """The thread id a rollout's file name ends with; None for a file that is not a rollout."""
    m = _ROLLOUT_NAME_RE.fullmatch(os.path.basename(str(path or "")))
    return m.group(4) if m else None


def codex_rollout_date_dir(name):
    """`YYYY/MM/DD`, the folder under `sessions` Codex files a rollout of this name in."""
    m = _ROLLOUT_NAME_RE.fullmatch(os.path.basename(str(name or "")))
    return os.path.join(m.group(1), m.group(2), m.group(3)) if m else None


def codex_history_logs(folder):
    """[(path, mtime, size)] for every rollout under `folder`, at any depth: Codex files them
    under `sessions/YYYY/MM/DD/` and moves an archived one to `archived_sessions/`. A rollout
    and its compressed copy are both listed (`.jsonl` first); size is on disk."""
    out = []
    for d, subdirs, names in os.walk(folder):
        subdirs.sort()
        for name in sorted(names):
            if codex_rollout_id(name) is None:
                continue
            path = os.path.join(d, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out.append((path, st.st_mtime, st.st_size))
    return out


def codex_rollout_head(path):
    """What a rollout's `session_meta` record says about its session: {"session_id", "cwd",
    "started" (epoch seconds), "source", "originator", "headless", "subagent"}, each None (or
    False) where it does not say. `headless`: run by `codex exec` (`source: "exec"`, or
    originator `codex_exec`, 0.163.0); `subagent`: a subagent's own rollout (`source:
    {"subagent": ...}`), whose work belongs to the session that spawned it."""
    out = {"session_id": None, "cwd": None, "started": None, "source": None, "originator": "",
           "headless": False, "subagent": False}
    for ev in head_records(path, HEAD_LINES, HEAD_BYTES, LINE_BYTES):
        if ev.get("type") != "session_meta":
            continue
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        sid = p.get("id") or p.get("session_id")
        out["session_id"] = sid if isinstance(sid, str) else None
        out["cwd"] = p.get("cwd") if isinstance(p.get("cwd"), str) and p.get("cwd") else None
        out["started"] = _epoch(p.get("timestamp")) or _epoch(ev.get("timestamp"))
        out["source"] = p.get("source")
        out["originator"] = str(p.get("originator") or "")
        out["headless"] = p.get("source") == "exec" or out["originator"] == "codex_exec"
        out["subagent"] = isinstance(p.get("source"), dict) and "subagent" in p["source"]
        break
    return out


def codex_prompt_turns(rollout_path):
    """Each prompt-turn of a Codex session, as its records from the record that opens it
    (`_turn_starts`) to the next, one turn at a time; records before the first are no turn."""
    starts = _turn_starts(rollout_path)
    current = None
    for d in _records(rollout_path):
        if starts(d):
            if current is not None:
                yield current
            current = [d]
        elif current is not None:
            current.append(d)
    if current is not None:
        yield current


def _events(turn, kind):
    return [p for d in turn if d.get("type") == "event_msg"
            for p in [d.get("payload") if isinstance(d.get("payload"), dict) else {}]
            if p.get("type") == kind]


def _shell_script(command):
    """The command the model ran, without the shell Codex wrapped it in (`bash -lc SCRIPT`,
    `pwsh.exe -Command SCRIPT`), as the live hooks see it."""
    if isinstance(command, list):
        argv = [str(c) for c in command]
        if len(argv) >= 3 and argv[-2].lower() in _SHELL_SCRIPT_FLAGS:
            return argv[-1]
        return shlex.join(argv)
    return str(command or "")


def _mcp_result_text(result):
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        for key in ("Err", "error", "text"):
            if isinstance(result.get(key), str):
                return result[key]
    return result if isinstance(result, str) else ""


def codex_turn_calls(turn):
    """[(name, args, result_text, exit_code_or_None)] for one prompt-turn's records, as the live
    hooks would have folded them: a command as `bash` with the script the model ran and the
    exit code Codex recorded; a file change as `apply_patch` with its paths; an image view and
    an MCP call by the names the hooks give them. The web search Codex runs itself is no hook
    call."""
    out = []
    for p in _events(turn, "item_completed"):
        item = p.get("item") if isinstance(p.get("item"), dict) else {}
        kind = item.get("type")
        if kind == "CommandExecution":
            output = item.get("aggregated_output") or ((item.get("stdout") or "")
                                                       + (item.get("stderr") or ""))
            code = item.get("exit_code")
            code = code if isinstance(code, int) and not isinstance(code, bool) else None
            out.append(("bash", {"command": _shell_script(item.get("command"))},
                        str(output or ""), code))
        elif kind == "FileChange":
            changes = item.get("changes")
            if isinstance(changes, dict):
                paths = [str(k) for k in changes]
            else:
                paths = [c.get("path") for c in changes or []
                         if isinstance(c, dict) and c.get("path")]
            out.append(("apply_patch", {"path": paths[0] if paths else "", "paths": paths},
                        "", None))
        elif kind == "ImageView":
            out.append(("view_image", {"path": str(item.get("path") or "")}, "", None))
        elif kind == "McpToolCall":
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            out.append((f"mcp__{item.get('server')}__{item.get('tool')}", args,
                        _mcp_result_text(item.get("result") or item.get("error")), None))
    return out


def codex_turn_interrupted(turn):
    """Did the person stop this turn? Codex then records a `turn_aborted` event."""
    return bool(_events(turn, "turn_aborted"))


def codex_turn_completed(turn):
    """Did this turn run to its end? Codex then records a `task_complete` event."""
    return bool(_events(turn, "task_complete"))



def codex_session_meta(rollout_path):
    """{cwd, model, cli_version} from a rollout's session_meta / turn_context records."""
    meta = {}
    for line in _lines(rollout_path):
        if '"session_meta"' not in line and '"turn_context"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if ev.get("type") == "session_meta":
            meta.setdefault("cwd", p.get("cwd"))
            meta.setdefault("cli_version", p.get("cli_version"))
        elif ev.get("type") == "turn_context" and p.get("model"):
            meta["model"] = p.get("model")
    return meta


def codex_session_usage(rollout_path):
    """Token breakdown of a Codex session from its last cumulative `token_count`.

    Converted as `te_harness.codex.row_usage` converts the exec stream: fresh input is the grand
    input total less both cache fields, and reasoning tokens are added to output."""
    total = None
    requests = 0
    for line in _lines(rollout_path):
        if '"token_count"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if p.get("type") != "token_count":
            continue
        info = p.get("info") or {}
        if info.get("total_token_usage"):
            total = info["total_token_usage"]
            requests += 1
    meta = codex_session_meta(rollout_path)
    if not total:
        return {"input_tokens": None, "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None, "output_tokens": None,
                "requests": 0, "model": meta.get("model")}
    inp = int(total.get("input_tokens") or 0)
    cached = int(total.get("cached_input_tokens") or 0)
    cw = int(total.get("cache_write_input_tokens") or 0)
    reasoning = int(total.get("reasoning_output_tokens") or 0)
    return {"input_tokens": max(0, inp - cached - cw),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": cw,
            "output_tokens": int(total.get("output_tokens") or 0) + reasoning,
            "reasoning_output_tokens": reasoning,
            "requests": requests, "model": meta.get("model")}
