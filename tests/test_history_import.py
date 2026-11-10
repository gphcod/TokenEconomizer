"""Importing a repository's sessions from before the plugin, from the agent's own logs.

Logs are laid out as Claude Code keeps them: `<config>/projects/<folder>/<session id>.jsonl`,
each entry naming the session, its directory and the time; and as Codex keeps its rollouts:
`<CODEX_HOME>/sessions/YYYY/MM/DD/rollout-<time>-<thread id>.jsonl[.zst]` and
`archived_sessions/`. The cases that are about time order
call the import and the fold directly, and stand in for the detached processes, rather than
running everything inline (`TE_FOLD_SYNC=1`), which would hide the races they are about.
"""
import io
import json
import os
import threading
import time
from datetime import datetime, timezone

import pytest

from test_live_hooks import _catalog, call, store_for, trap_session  # noqa: F401
from tokeneconomizer import trace
from tokeneconomizer.plugin import (accounting, cli, denylist, fold, history, hook, hosts, repos,
                                    sidecall)
from tokeneconomizer.plugin.config import load_config
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.paths import repo_identity

CLAIM = "`pytest …` fails here with command not found; `python -m pytest …` works."
UNUSED = ["CronCreate", "ShareOnboardingGuide", "Workflow"]


@pytest.fixture
def claude_home(tmp_path):
    d = tmp_path / "claude"
    (d / "projects").mkdir(parents=True)
    return d


@pytest.fixture
def codex_home(tmp_path):
    d = tmp_path / "codex"
    (d / "sessions").mkdir(parents=True)
    return d


@pytest.fixture
def env(tmp_path, claude_home, codex_home):
    """A person who turned the import on (`history_import` is off by default)."""
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_HISTORY_IMPORT": "on",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "CLAUDE_CONFIG_DIR": str(claude_home), "CODEX_HOME": str(codex_home),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    (r / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(r)


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def trap(repo, i):
    """The prompts of a session that meets the missing `pytest` and works around it."""
    return [(f"fix bug {i}", [
        ("Edit", {"file_path": f"{repo}/pkg/mod{i}.py", "old_string": "a", "new_string": "b"},
         "ok", False),
        ("Bash", {"command": "pytest -q tests/test_a.py"}, "Exit code 127\nbash: pytest: command "
                                                          "not found", True),
        ("Bash", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s", False),
    ])]


def looked(repo):
    return [("look", [("Read", {"file_path": f"{repo}/a.py"}, "x", False),
                      ("Bash", {"command": "ls"}, "a.py", False)]), ("and again", [])]


def history_log(claude_home, cwd, sid, prompts, age_s=3600, folder=None, entrypoint="cli",
                deferred=(), interrupted=False):
    """A finished session's log where Claude Code keeps it, last written `age_s` ago. The
    person interrupts the prompts at the indices `interrupted` names (True: the last one)."""
    cut = {len(prompts) - 1} if interrupted is True else set(interrupted or ())
    folder = folder or trace.claude_project_folder(cwd)
    d = os.path.join(str(claude_home), "projects", folder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    end = time.time() - age_s
    t = end - 300
    base = {"sessionId": sid, "cwd": cwd, "entrypoint": entrypoint, "version": "2.1.307"}
    lines, n = [{"type": "file-history-snapshot", "snapshot": {}}], 0
    if deferred:
        lines.append(dict(base, type="attachment", timestamp=_iso(t), attachment={
            "type": "deferred_tools_delta", "addedNames": list(deferred)}))
    for k, (text, calls) in enumerate(prompts):
        t += 1
        lines.append(dict(base, type="user", timestamp=_iso(t),
                          message={"role": "user", "content": text}))
        for name, inp, result, is_error in calls:
            n += 1
            lines.append(dict(base, type="assistant", timestamp=_iso(t), message={
                "id": f"msg_{sid}_{n}", "model": "claude-opus-5-5", "role": "assistant",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                          "cache_creation_input_tokens": 100, "output_tokens": 50},
                "content": [{"type": "tool_use", "id": f"tu{n}", "name": name, "input": inp}]}))
            lines.append(dict(base, type="user", timestamp=_iso(t), message={
                "role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu{n}",
                                             "content": result, "is_error": is_error}]}))
        if k in cut:
            lines.append(dict(base, type="user", timestamp=_iso(t), message={
                "role": "user", "content": [{"type": "text",
                                             "text": "[Request interrupted by user]"}]}))
        else:
            lines.append(dict(base, type="assistant", timestamp=_iso(t), message={
                "id": f"msg_{sid}_{n}_done", "model": "claude-opus-5-5", "role": "assistant",
                "content": [{"type": "text", "text": "Done."}]}))
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    os.utime(path, (end, end))
    return path


def ctx_for(env, repo):
    return LiveContext.build(hosts.ClaudeCode(), repo, env)


def detached(env):
    """The environment of a plugin process that hands its work to a detached one."""
    return {k: v for k, v in env.items() if k != "TE_FOLD_SYNC"}


def live_session(ctx, sid, transcript, started=None):
    """A session the plugin's hooks recorded, ended and not yet folded."""
    now = time.time()
    ctx.store.write(os.path.join("sessions", f"{sid}.json"), {
        "session_id": sid, "host": "claude", "started": started or now, "ended": now,
        "cwd": ctx.repo.root, "transcript": transcript, "turns": []})


def start_hook(env, repo, sid, tmp_path):
    t = str(tmp_path / f"{sid}.jsonl")
    open(t, "w").close()
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid,
                      "source": "startup", "transcript_path": t, "cwd": repo})


def ledger_ids(store):
    return [r["session_id"] for r in store.read_lines("ledger.jsonl") if r["kind"] == "session"]


class Died(BaseException):
    """The import's process killed mid-session (out of memory, a signal): nothing it catches."""


# ---- what is read, and how ------------------------------------------------------------------
def test_the_first_session_start_reads_the_repositorys_history_once_oldest_first(
        env, repo, claude_home, tmp_path):
    for i in range(4):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    other = tmp_path / "other"
    other.mkdir()
    history_log(claude_home, str(other), "elsewhere", trap(str(other), 0))
    history_log(claude_home, repo, "running", trap(repo, 9), age_s=5)      # still writing
    history_log(claude_home, repo, "scripted", trap(repo, 8), entrypoint="sdk-cli")
    history_log(claude_home, repo, "empty", [])
    sub = os.path.join(str(claude_home), "projects", trace.claude_project_folder(repo), "old0",
                       "subagents")
    os.makedirs(sub)
    with open(os.path.join(sub, "agent-x.jsonl"), "w") as fh:
        fh.write("{}\n")

    first = start_hook(env, repo, "new1", tmp_path)
    st = store_for(env, repo)
    assert ledger_ids(st) == ["old0", "old1", "old2", "old3"]
    assert all(r["imported"] and r["untool_applied"] == [] for r in st.read_lines("ledger.jsonl"))
    assert all(r["overhead"]["optimizer_tokens"] == 0 for r in st.read_lines("ledger.jsonl"))
    assert all(e["imported"] for e in st.read("folded.json").values())
    doc = st.read("history.json")
    assert doc["state"] == "done" and doc["imported"] == 4 and "plan" not in doc
    assert doc["skipped"] == {"recent": 1, "headless": 1, "no_prompt": 1}
    assert "systemMessage" not in first                 # said at the next start

    second = start_hook(env, repo, "new2", tmp_path)
    assert CLAIM in second["hookSpecificOutput"]["additionalContext"]
    assert ("TokenEconomizer read 4 earlier session(s) of acme/widget from Claude Code's "
            "history.") in second["systemMessage"]
    assert "earlier session" not in start_hook(env, repo, "new3", tmp_path).get(
        "systemMessage", "")


def test_history_import_off_reads_nothing_and_the_newest_are_kept(env, repo, claude_home,
                                                                   tmp_path):
    for i in range(4):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    history_log(claude_home, repo, "ancient", trap(repo, 7), age_s=400 * 86400)
    start_hook(dict(env, TE_HISTORY_IMPORT="off"), repo, "a", tmp_path)
    st = store_for(env, repo)
    assert st.read("history.json") is None and not st.read("folded.json")
    capped = dict(env, TE_HISTORY_MAX_SESSIONS="2")
    start_hook(capped, repo, "b", tmp_path)
    assert ledger_ids(st) == ["old2", "old3"]
    doc = st.read("history.json")
    assert doc["over_limit"] == 2 and doc["skipped"]["too_old"] == 1


def test_by_default_a_session_start_reads_no_history_and_an_import_by_hand_does(
        env, repo, claude_home, tmp_path, monkeypatch, capsys):
    """Off by default: nothing is read until the person asks, by the setting or by hand."""
    assert load_config(str(tmp_path / "empty"), env={}).history_import is False
    for i in range(2):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    default = {k: v for k, v in env.items() if k != "TE_HISTORY_IMPORT"}
    start_hook(default, repo, "a", tmp_path)
    st = store_for(env, repo)
    assert st.read("history.json") is None and not st.read("folded.json")

    for k, v in default.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("TE_HISTORY_IMPORT", raising=False)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 0
    assert "imported 2 earlier session(s)" in capsys.readouterr().out
    assert ledger_ids(st) == ["old0", "old1"]


def test_a_moved_checkouts_logs_are_matched_by_their_exact_folder(env, repo, claude_home):
    """Claude Code files a moved directory's logs under the folder of its new path, while they
    still name the old one. The folder must match exactly: `widget-shell` is not `widget`."""
    folder = trace.claude_project_folder(repo)
    history_log(claude_home, "/gone/old/place/repo", "moved", trap(repo, 1), folder=folder)
    history_log(claude_home, "/gone/old/place/repo-shell", "shell", trap(repo, 2),
                folder=folder + "-shell")
    history_log(claude_home, "/gone/elsewhere", "prefix", trap(repo, 3), folder=folder[:-2])
    ctx = ctx_for(env, repo)
    assert [s["id"] for s in history.plan(ctx)["sessions"]] == ["moved"]


def test_unsafe_session_ids_are_never_used(env, repo, claude_home):
    for bad in ("x$(touch y)", "-rf", "a.b"):
        history_log(claude_home, repo, bad, trap(repo, 1))
    history_log(claude_home, repo, "0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b7a", trap(repo, 2))
    p = history.plan(ctx_for(env, repo))
    assert [s["id"] for s in p["sessions"]] == ["0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b7a"]
    assert p["skipped"] == {"unsafe_id": 3}
    for bad in ("../x", "a/b", "", None, "x" * 200, "a\\b"):
        assert trace.claude_safe_session_id(bad) is None


# ---- time order ------------------------------------------------------------------------------
def test_a_repository_with_folded_sessions_records_the_skip_and_a_manual_import_refuses(
        env, repo, claude_home, tmp_path, monkeypatch, capsys):
    """An upgrade: the plugin has already folded this repository's sessions, so the sessions
    from before it would be learned after them."""
    off = dict(env, TE_HISTORY_IMPORT="off")
    for i in range(2):
        t = trap_session(str(tmp_path / f"s{i}.jsonl"), repo, i)
        call(off, {"hook_event_name": "SessionStart", "session_id": f"s{i}",
                   "source": "startup", "transcript_path": t, "cwd": repo})
        call(off, {"hook_event_name": "SessionEnd", "session_id": f"s{i}",
                   "transcript_path": t, "cwd": repo})
    history_log(claude_home, repo, "old0", trap(repo, 0))
    st = store_for(env, repo)
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    start_hook(detached(env), repo, "s2", tmp_path)
    doc = st.read("history.json")
    assert doc["state"] == "skipped" and "already" in doc["reason"] and spawned == []
    start_hook(detached(env), repo, "s3", tmp_path)
    assert st.read("history.json") == doc and spawned == []            # never retried

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 1
    out = capsys.readouterr().out.strip()
    assert out.startswith("not imported: acme/widget already has folded sessions")
    assert len(out.splitlines()) == 1
    assert "old0" not in st.read("folded.json")


def test_a_session_that_ends_during_the_import_is_folded_after_it(env, repo, claude_home,
                                                                     tmp_path, monkeypatch):
    for i in range(3):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    ctx = ctx_for(detached(env), repo)
    history.maybe_start(ctx)                    # the first session start: detached
    assert spawned == [["history", "import", "--auto"]]
    assert ctx.store.read("history.json")["state"] == "running"

    # That session ends before the detached import has even planned: its fold waits.
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    assert fold.fold_session(ctx, "live0") is None
    assert fold.spawn_fold(ctx, ["live0"]) is None and spawned == [["history", "import",
                                                                    "--auto"]]
    # Another ends while the import holds the lock, between two imported sessions.
    real = fold.fold_locked
    attempts = []

    def fold_and_race(c, sid, *a, **kw):
        rec = real(c, sid, *a, **kw)
        if sid == "old0":
            live_session(ctx, "live1", trap_session(str(tmp_path / "live1.jsonl"), repo, 6))
            t0 = time.monotonic()
            attempts.append(fold.fold_session(ctx_for(detached(env), repo), "live1"))
            assert time.monotonic() - t0 < 5        # never waits on the import's lock
        return rec
    monkeypatch.setattr(fold, "fold_locked", fold_and_race)
    res = history.run(ctx_for(detached(env), repo), auto=True)
    assert res["state"] == "done" and res["imported"] == 3 and attempts == [None]
    assert ledger_ids(ctx.store) == ["old0", "old1", "old2", "live0", "live1"]
    assert not ctx.store.read("folded.json")["live0"].get("imported")


def test_a_run_stopped_by_its_budget_resumes_the_same_plan_in_order(env, repo, claude_home,
                                                                     tmp_path, monkeypatch):
    for i in range(4):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    env = dict(env, TE_HISTORY_MAX_SESSIONS="3")
    ctx = ctx_for(detached(env), repo)
    clock = iter(range(1000))
    monkeypatch.setattr(history, "_clock", lambda: next(clock))
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    res = history.run(ctx, auto=True, budget_s=2.5)
    assert res["state"] == "partial" and res["imported"] == 2
    doc = ctx.store.read("history.json")
    assert [s["id"] for s in doc["plan"]] == ["old1", "old2", "old3"] and doc["next"] == 2
    assert fold.import_blocks(ctx)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    assert fold.fold_session(ctx, "live0") is None          # still waits for the rest
    # That session paid for the loop guard's judge meanwhile: a charge, not a fold.
    ctx.store.append("ledger.jsonl", {"kind": "overhead", "session_id": "live0", "host": "claude",
                                      "at": time.time(), "overhead": {"spiral_judge_tokens": 9.0}})

    # A newer log since: the resumed run keeps the plan it made, so nothing is skipped over.
    history_log(claude_home, repo, "old4", trap(repo, 4), age_s=1800)
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    history.maybe_start(ctx)
    assert spawned == [["history", "import", "--auto"]]
    res = history.run(ctx_for(detached(env), repo), auto=True, budget_s=100)
    assert res["state"] == "done" and res["imported"] == 3
    assert ledger_ids(ctx.store) == ["old1", "old2", "old3", "live0"]


def test_a_dead_import_is_restarted_by_the_next_session_start(env, repo, claude_home,
                                                               monkeypatch):
    history_log(claude_home, repo, "old0", trap(repo, 0))
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    history.maybe_start(ctx)
    assert spawned == []                                    # alive: left to run
    ctx.store.write("history.json", {"state": "running",
                                     "heartbeat": time.time() - history.HEARTBEAT_STALE_S - 1})
    history.maybe_start(ctx)
    assert spawned == [["history", "import", "--auto"]]
    assert time.time() - ctx.store.read("history.json")["heartbeat"] < 60
    history.maybe_start(ctx)
    assert len(spawned) == 1                                # one restart, not one per start
    assert history.run(ctx, auto=True)["state"] == "done"
    assert "old0" in ctx.store.read("folded.json")

    # With the import turned off meanwhile, a stopped run stops blocking live folds.
    ctx.store.write("history.json", {"state": "partial", "plan": [], "next": 0})
    off = ctx_for(dict(detached(env), TE_HISTORY_IMPORT="off"), repo)
    history.maybe_start(off)
    assert ctx.store.read("history.json")["state"] == "stopped"
    assert not fold.import_blocks(ctx)


def test_an_import_that_fails_stops_blocking_live_folds(env, repo, claude_home, tmp_path,
                                                        monkeypatch):
    history_log(claude_home, repo, "old0", trap(repo, 0))
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})

    def broken(*a, **k):
        raise RuntimeError("unreadable history")
    monkeypatch.setattr(history, "plan", broken)
    assert history.run(ctx, auto=True)["state"] == "error"
    assert ctx.store.read("history.json")["state"] == "stopped" and not fold.import_blocks(ctx)
    assert "unreadable history" in open(os.path.join(ctx.root, "errors.log")).read()
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    assert fold.fold_session(ctx, "live0") is not None


def test_an_import_waits_for_a_live_fold_already_under_way(env, repo, claude_home, tmp_path):
    """A session's end that checked `history.json` just before the first session start wrote
    `running` folds under the lock; the import waits for it, then finds a folded session."""
    history_log(claude_home, repo, "old0", trap(repo, 0))
    ctx = ctx_for(detached(env), repo)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    holding = threading.Event()

    def live_fold():
        c = ctx_for(detached(env), repo)
        with c.store.locked("fold") as got:
            holding.set()
            time.sleep(0.5)
            if got:
                fold.fold_locked(c, "live0")
    th = threading.Thread(target=live_fold)
    th.start()
    assert holding.wait(10)
    t0 = time.monotonic()
    res = history.run(ctx_for(detached(env), repo), auto=True)
    th.join()
    assert time.monotonic() - t0 >= 0.3                     # it waited for the lock
    assert res["state"] == "skipped" and "already has folded sessions" in res["reason"]
    assert ledger_ids(ctx.store) == ["live0"]
    assert ctx.store.read("history.json")["state"] == "skipped" and not fold.import_blocks(ctx)


def test_a_live_fold_that_waited_on_the_lock_looks_at_the_import_again(env, repo, tmp_path,
                                                                        monkeypatch):
    """It passed the check before the lock; meanwhile an import took the lock and released it
    `partial`. Under the lock it must see that and leave the session pending."""
    ctx = ctx_for(detached(env), repo)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    real, seen = fold.import_blocks, []

    def import_runs_meanwhile(c):
        seen.append(1)
        if len(seen) == 1:
            c.store.write("history.json", {"state": "partial", "plan": [], "next": 0})
            return False
        return real(c)
    monkeypatch.setattr(fold, "import_blocks", import_runs_meanwhile)
    assert fold.fold_session(ctx, "live0") is None and len(seen) == 2
    assert "live0" not in (ctx.store.read("folded.json") or {})
    assert fold.pending_sessions(ctx) == ["live0"]


def test_what_a_plan_leaves_out(env, repo, claude_home, monkeypatch):
    small = history_log(claude_home, repo, "small", trap(repo, 0), age_s=7200)
    history_log(claude_home, repo, "large", trap(repo, 1) + looked(repo), age_s=5400)
    history_log(claude_home, repo, "seen", trap(repo, 2), age_s=3600)
    ctx = ctx_for(env, repo)
    ctx.store.write(os.path.join("sessions", "seen.json"), {"session_id": "seen"})
    monkeypatch.setattr(history, "MAX_BYTES", os.path.getsize(small))
    p = history.plan(ctx)
    assert [s["id"] for s in p["sessions"]] == ["small"]
    assert p["skipped"] == {"too_large": 1, "recorded_live": 1}


def test_folded_means_a_session_record_not_a_charge(env, repo, claude_home, monkeypatch):
    """A ledger that holds a session's record is folded state though `folded.json` is empty; a
    ledger that holds only the loop guard's judge's charges is not."""
    history_log(claude_home, repo, "old0", trap(repo, 0))
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    ctx = ctx_for(detached(env), repo)
    ctx.store.append("ledger.jsonl", {"kind": "overhead", "session_id": "live0", "at": 1.0,
                                      "overhead": {"spiral_judge_tokens": 9.0}})
    history.maybe_start(ctx)
    assert spawned == [["history", "import", "--auto"]]
    assert history.run(ctx, auto=True)["state"] == "done"

    other = ctx_for(detached(dict(env, TOKENECONOMIZER_HOME=env["TOKENECONOMIZER_HOME"] + "2")),
                    repo)
    other.store.append("ledger.jsonl", {"kind": "session", "session_id": "s0", "folded_at": 1.0})
    history.maybe_start(other)
    assert spawned == [["history", "import", "--auto"]]
    assert other.store.read("history.json")["state"] == "skipped"


def test_a_session_whose_fold_fails_is_skipped_and_the_rest_imported(env, repo, claude_home,
                                                                     monkeypatch):
    for i in range(3):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    real = fold.fold_locked

    def fails_on_old1(c, sid, *a, **kw):
        if sid == "old1":
            raise RuntimeError("a log it cannot fold")
        return real(c, sid, *a, **kw)
    monkeypatch.setattr(fold, "fold_locked", fails_on_old1)
    res = history.run(ctx, auto=True)
    assert res["state"] == "done" and res["imported"] == 2 and res["skipped"] == {"error": 1}
    assert ledger_ids(ctx.store) == ["old0", "old2"]


def test_a_session_the_import_dies_on_is_left_out_after_two_tries(env, repo, claude_home,
                                                                  monkeypatch):
    """The process dies (out of memory, a signal) while it folds one log: each restart tries
    again, but not for ever, or live folds would wait for good."""
    for i in range(2):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    real = fold.fold_locked

    def dies_on_old0(c, sid, *a, **kw):
        if sid == "old0":
            raise Died()
        return real(c, sid, *a, **kw)
    monkeypatch.setattr(fold, "fold_locked", dies_on_old0)
    for n in (1, 2):
        with pytest.raises(Died):
            history.run(ctx, auto=True)
        doc = ctx.store.read("history.json")
        assert doc["state"] == "running" and doc["next"] == 0
        assert doc["attempt"] == {"id": "old0", "n": n}
    res = history.run(ctx, auto=True)
    assert res["state"] == "done" and res["imported"] == 1 and res["skipped"] == {"error": 1}
    assert ledger_ids(ctx.store) == ["old1"] and not fold.import_blocks(ctx)


def test_a_session_folded_just_before_the_import_died_counts_as_imported(env, repo, claude_home,
                                                                         monkeypatch):
    for i in range(2):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    real, died = fold.fold_locked, []

    def dies_after_old0(c, sid, *a, **kw):
        rec = real(c, sid, *a, **kw)
        if sid == "old0" and not died:
            died.append(1)
            raise Died()                    # folded; its place in the plan is not saved
        return rec
    monkeypatch.setattr(fold, "fold_locked", dies_after_old0)
    with pytest.raises(Died):
        history.run(ctx, auto=True)
    assert ctx.store.read("history.json")["next"] == 0
    res = history.run(ctx, auto=True)
    assert res["state"] == "done" and res["imported"] == 2 and res["skipped"] == {}
    assert ledger_ids(ctx.store) == ["old0", "old1"]
    assert ctx.store.read("envelope.json")["resolved_turns"] == [3, 3]


def test_te_history_import_resumes_a_plan_that_was_stopped(env, repo, claude_home, monkeypatch,
                                                           capsys):
    for i in range(2):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    ctx = ctx_for(detached(env), repo)
    clock = iter(range(1000))
    monkeypatch.setattr(history, "_clock", lambda: next(clock))
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True, budget_s=1.5)["state"] == "partial"
    doc = ctx.store.read("history.json")
    ctx.store.write("history.json", dict(doc, state="stopped", reason="history_import is off"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(history, "_clock", time.monotonic)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 0
    assert "imported 2 earlier session(s)" in capsys.readouterr().out
    assert ledger_ids(ctx.store) == ["old0", "old1"]


# ---- learning only ---------------------------------------------------------------------------
def test_imported_sessions_are_scored_against_their_own_tools(env, repo, claude_home):
    """A tool only in today's catalog did not exist for an earlier session: it is not counted
    as unused there. Tools a log listed (deferred) are."""
    _catalog(env, ["Bash", "Read", "Edit", "Write", "NewlyInstalled"])
    for i in range(3):
        history_log(claude_home, repo, f"old{i}", looked(repo), age_s=3600 * (10 - i),
                    deferred=["CronCreate"])
    ctx = ctx_for(env, repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    history.run(ctx, auto=True)
    unt = ctx.store.read("untool.json")
    assert "NewlyInstalled" not in unt["streaks"] and "NewlyInstalled" not in unt["catalog_tools"]
    assert unt["streaks"]["CronCreate"] == 3 and unt["imported_sessions"] == 3


def test_the_next_live_fold_under_auto_holds_nothing_that_rests_on_history(
        env, repo, claude_home, tmp_path, monkeypatch, capsys):
    _catalog(env, ["Bash", "Read", "Edit", "Write", "AskUserQuestion"] + UNUSED)
    for i in range(6):
        history_log(claude_home, repo, f"old{i}", looked(repo), age_s=3600 * (10 - i),
                    deferred=UNUSED)
    start_hook(env, repo, "first", tmp_path)
    st = store_for(env, repo)
    assert st.read("history.json")["imported"] == 6
    unt = st.read("untool.json")
    assert sorted(unt["proposal"]) == UNUSED and "auto_pending" not in unt
    assert sorted(unt["needs_apply"]) == UNUSED

    # The next live fold: plenty of evidence, but from history, so `auto` waits for the person.
    ctx = ctx_for(detached(env), repo)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 1))
    rec = fold.fold_session(ctx, "live0")
    assert rec is not None and not rec["untool"]["pending"]
    unt = st.read("untool.json")
    assert "auto_pending" not in unt and sorted(unt["needs_apply"]) == UNUSED
    assert not os.path.exists(denylist.settings_path(repo))
    notice = start_hook(env, repo, "second", tmp_path)["systemMessage"]
    assert "3 tool(s) went unused" in notice and "counting sessions from before" in notice
    assert "nothing changes until you run `/tokeneconomizer:toolprune apply`" in notice

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cli.main(["--cwd", repo, "--host", "claude", "toolprune", "show"])
    assert "waiting for `toolprune apply`" in capsys.readouterr().out

    # The plugin's own sessions, enough evidence alone: `auto` writes the list as before.
    for i in range(1, 6):
        live_session(ctx, f"live{i}", trap_session(str(tmp_path / f"live{i}.jsonl"), repo, i))
        fold.fold_session(ctx, f"live{i}")
    deny = json.load(open(denylist.settings_path(repo)))["permissions"]["deny"]
    assert sorted(deny) == UNUSED


def test_an_import_calls_no_model_and_keeps_a_waiting_deny_list(env, repo, claude_home,
                                                                tmp_path, monkeypatch):
    for i in range(3):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    asked = []
    monkeypatch.setattr(sidecall, "runner_for", lambda *a, **k: asked.append(a) or None)
    env = dict(env, TE_MICROSKILLS="on", TE_MICROSKILLS_PROPOSE_EVERY="1")
    ctx = ctx_for(env, repo)
    pending = {"items": ["Workflow"], "root": repo, "since": 1.0}
    ctx.store.write("untool.json", {"auto_pending": pending})
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True)["imported"] == 3 and asked == []
    assert ctx.store.read("untool.json")["auto_pending"] == pending
    assert ctx.store.read("injection.json")["imported_sessions"] == 3
    live_session(ctx, "seen", trap_session(str(tmp_path / "seen.jsonl"), repo, 7))
    fold.fold_session(ctx, "seen")
    assert asked                                # a session the plugin saw end still may


def test_an_import_ignores_and_keeps_the_deny_list_in_place(env, repo, claude_home):
    """Sessions from before the plugin ran under no deny list of its, and an import gives back
    no rule -- not even one for a tool that has since become protected."""
    env = dict(env, TE_TOOLPRUNE_PROTECT="Workflow")
    _catalog(env, ["Bash", "Read", "Edit", "Write"] + UNUSED)
    ctx = ctx_for(env, repo)
    assert denylist.apply(repo, ["Workflow", "CronCreate"], ctx.store) == ["Workflow",
                                                                           "CronCreate"]
    path = denylist.settings_path(repo)
    before = open(path).read()
    for i in range(2):
        history_log(claude_home, repo, f"old{i}", looked(repo), age_s=3600 * (10 - i),
                    deferred=UNUSED)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True)["imported"] == 2
    assert all(r["untool_applied"] == [] for r in ctx.store.read_lines("ledger.jsonl"))
    assert ctx.store.read("untool.json")["streaks"]["CronCreate"] == 2
    assert open(path).read() == before
    assert sorted(denylist.applied_for(ctx.store, repo)) == ["CronCreate", "Workflow"]


def test_the_envelope_is_seeded_from_turns_that_ended_green(env, repo, claude_home):
    history_log(claude_home, repo, "old0", trap(repo, 0) + looked(repo), age_s=7200)
    history_log(claude_home, repo, "old1", looked(repo)[:1], age_s=3600, interrupted=True)
    one_call = [("one", [("Read", {"file_path": f"{repo}/b.py"}, "y", False)])]
    history_log(claude_home, repo, "old2", one_call + trap(repo, 2), age_s=1800, interrupted=[0])
    ctx = ctx_for(env, repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    history.run(ctx, auto=True)
    # old0: its turns with calls (the empty "and again" has none); old1: interrupted; old2:
    # its one-call turn was interrupted, though another prompt followed it.
    assert ctx.store.read("envelope.json")["resolved_turns"] == [3, 2, 3]
    assert not os.path.exists(ctx.store.path("events.jsonl"))       # nothing was redirected


# ---- what is shown ---------------------------------------------------------------------------
def test_imported_sessions_are_counted_apart(env, repo, claude_home, tmp_path, monkeypatch,
                                             capsys):
    s = accounting.summarize([
        {"kind": "session", "effective_tokens": 100.0, "requests": 2},
        {"kind": "session", "effective_tokens": 900.0, "requests": 9, "imported": True,
         "overhead": {"optimizer_tokens": 50}}])
    assert s["sessions"] == 1 and s["imported_sessions"] == 1
    assert s["effective_tokens"] == 100.0 and s["requests"] == 2 and s["overhead_tokens"] == 0

    for i in range(2):
        history_log(claude_home, repo, f"old{i}", trap(repo, i), age_s=3600 * (10 - i))
    start_hook(env, repo, "new", tmp_path)
    call(env, {"hook_event_name": "SessionEnd", "session_id": "new",
               "transcript_path": trap_session(str(tmp_path / "new.jsonl"), repo, 5),
               "cwd": repo})
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cli.main(["--cwd", repo, "--host", "claude", "status"])
    out = capsys.readouterr().out
    assert "sessions     1 folded (+2 from earlier history)" in out
    assert "(1 sessions learned from, +2 from earlier history)" in out
    assert "read 2 earlier session(s) from Claude Code's history" in out
    cli.main(["--cwd", repo, "--host", "claude", "status", "--json"])
    js = json.loads(capsys.readouterr().out)
    assert js["toolprune"]["sessions"] == 1 and js["toolprune"]["imported_sessions"] == 2
    assert js["ledger"]["sessions"] == 1 and js["history"]["imported"] == 2
    cli.main(["--cwd", repo, "--host", "claude", "repos"])
    assert "claude 1 (+2 from earlier history)" in capsys.readouterr().out


def test_a_merge_carries_imported_sessions_along(env, repo, claude_home, tmp_path):
    history_log(claude_home, repo, "old0", trap(repo, 0))
    start_hook(env, repo, "new", tmp_path)
    root = env["TOKENECONOMIZER_HOME"]
    src = repo_identity(repo, env).key
    other = tmp_path / "other"
    other.mkdir()
    dst = repo_identity(str(other), env)
    os.makedirs(os.path.join(repos.repos_dir(root), dst.key, "claude"))
    assert repos.merge(root, src, dst) is not None
    st = store_for(env, str(other))
    assert st.read("folded.json")["old0"]["imported"]
    assert [r for r in st.read_lines("ledger.jsonl") if r.get("imported")]
    assert st.read("history.json")["state"] == "done"


def test_a_merge_into_a_repository_that_learned_more_does_not_fold_imported_sessions_again(
        env, repo, claude_home, tmp_path):
    """The destination keeps its own learned state (more sessions folded): what it never saw
    is folded again from its log, but not a session imported from history, which would then
    land after the live ones."""
    history_log(claude_home, repo, "old0", trap(repo, 0))
    start_hook(env, repo, "new", tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    off = dict(env, TE_HISTORY_IMPORT="off")
    for i in range(2):
        t = trap_session(str(tmp_path / f"d{i}.jsonl"), str(other), i)
        call(off, {"hook_event_name": "SessionStart", "session_id": f"d{i}",
                   "source": "startup", "transcript_path": t, "cwd": str(other)})
        call(off, {"hook_event_name": "SessionEnd", "session_id": f"d{i}",
                   "transcript_path": t, "cwd": str(other)})
    st = store_for(env, str(other))
    assert sorted(st.read("folded.json")) == ["d0", "d1"]
    root = env["TOKENECONOMIZER_HOME"]
    res = repos.merge(root, repo_identity(repo, env).key, repo_identity(str(other), env))
    assert res["claude"]["refold"] == 0
    assert st.read("folded.json")["old0"]["imported"]
    assert not st.read(os.path.join("sessions", "old0.json"))


def test_a_malformed_history_file_never_empties_the_session_start(env, repo, tmp_path,
                                                                   monkeypatch):
    for i in range(6):
        t = trap_session(str(tmp_path / f"s{i}.jsonl"), repo, i)
        call(env, {"hook_event_name": "SessionStart", "session_id": f"s{i}",
                   "source": "startup", "transcript_path": t, "cwd": repo})
        call(env, {"hook_event_name": "SessionEnd", "session_id": f"s{i}",
                   "transcript_path": t, "cwd": repo})
    st = store_for(env, repo)
    path = st.path("history.json")
    for bad in ('{"state": "done", "noticed": false, "imported": {"x": 1}}',
                '{"state": "done", "noticed": false, "imported": "many"}',
                '{"state": ["running"], "heartbeat": "soon"}', '[1, 2]', '{not json'):
        with open(path, "w") as fh:
            fh.write(bad)
        out = start_hook(env, repo, "n", tmp_path)
        assert CLAIM in out["hookSpecificOutput"]["additionalContext"], bad
        assert not fold.import_blocks(ctx_for(env, repo))

    def boom(*a, **k):
        raise ValueError("a field history.json should not have")
    monkeypatch.setattr(history, "notice", boom)
    monkeypatch.setattr(history, "maybe_start", boom)
    out = start_hook(env, repo, "n2", tmp_path)
    assert CLAIM in out["hookSpecificOutput"]["additionalContext"]
    log = open(os.path.join(env["TOKENECONOMIZER_HOME"], "errors.log")).read()
    assert log.count("ValueError: a field history.json should not have") == 2


def test_the_history_settings(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(
        {"history_dirs": ["/a", "/b"], "history_import": "no", "history_max_sessions": 50}))
    cfg = load_config(str(root), env={})
    assert cfg.history_dirs == ["/a", "/b"] and cfg.history_import is False
    assert cfg.history_max_sessions == 50 and cfg.history_days == 365
    cfg = load_config(str(root), env={"TE_HISTORY_IMPORT": "on", "TE_HISTORY_DAYS": "30",
                                      "TE_HISTORY_MAX_SESSIONS": "0"})
    assert cfg.history_import is True and cfg.history_days == 30
    assert cfg.history_max_sessions == 50 and any("history_max_sessions" in w
                                                  for w in cfg.warnings)
    view = cli._config_view(load_config(str(root), env={}))
    assert view["history"]["state"] == "off" and view["history"]["history_dirs"] == ["/a", "/b"]


def test_where_claude_code_keeps_its_logs(tmp_path):
    assert trace.claude_projects_dir({"CLAUDE_CONFIG_DIR": "/c"}) == os.path.join("/c",
                                                                                   "projects")
    assert trace.claude_project_folder("/home/u/my_repo.v2") == "-home-u-my-repo-v2"
    proj = tmp_path / "projects" / "-w"
    (proj / "s1" / "subagents").mkdir(parents=True)
    for p in (proj / "s1.jsonl", proj / "agent-a.jsonl", proj / "s1" / "subagents" / "x.jsonl"):
        p.write_text("{}\n")
    logs = hosts.ClaudeCode().history({"CLAUDE_CONFIG_DIR": str(tmp_path / "none")},
                                      [str(tmp_path)])
    assert [(x["session_id"], x["folder"]) for x in logs] == [("s1", "-w")]
    assert hosts.Codex().history({"CODEX_HOME": str(tmp_path / "none")}) == []


def test_te_history_dry_run_writes_nothing(env, repo, claude_home, monkeypatch, capsys):
    history_log(claude_home, repo, "old0", trap(repo, 0))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--dry-run"]) == 0
    assert "1 earlier session(s) of acme/widget would be read" in capsys.readouterr().out
    assert store_for(env, repo).read("history.json") is None
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 0
    assert "imported 1 earlier session(s)" in capsys.readouterr().out
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--status"]) == 0
    assert "read 1 earlier session(s)" in capsys.readouterr().out
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 1


def test_no_hook_reads_another_agents_history(env, repo, claude_home, tmp_path):
    """Codex's import reads Codex's rollouts, never Claude Code's logs; an agent that keeps no
    history the plugin reads (Copilot) imports nothing."""
    history_log(claude_home, repo, "old0", trap(repo, 0))
    t = str(tmp_path / "rollout-x.jsonl")
    open(t, "w").close()
    out = io.StringIO()
    hook.main(["--host", "codex"], io.StringIO(json.dumps({
        "hook_event_name": "SessionStart", "session_id": "c1", "source": "startup",
        "transcript_path": t, "cwd": repo})), out, env)
    st = store_for(env, repo, host="codex")
    assert st.read("history.json")["imported"] == 0 and not st.read("folded.json")
    hook.main(["--host", "copilot"], io.StringIO(json.dumps({
        "hook_event_name": "SessionStart", "session_id": "p1", "source": "startup",
        "cwd": repo})), io.StringIO(), env)
    assert store_for(env, repo, host="copilot").read("history.json") is None


# ---- Codex ----------------------------------------------------------------------------------
try:
    from compression import zstd as _zstd
except ImportError:                     # Python < 3.14
    _zstd = None
needs_zstd = pytest.mark.skipif(_zstd is None, reason="this Python has no compression.zstd")


def cx(n):
    """A thread id, as Codex names one."""
    return f"01a1c000-0000-7000-8000-{n:012d}"


def cx_trap(i):
    """A Codex session's prompt that meets the missing `pytest` and works around it."""
    return [(f"fix bug {i}", [
        ("patch", [f"pkg/mod{i}.py"]),
        ("cmd", "pytest -q tests/test_a.py", "bash: pytest: command not found", 127),
        ("cmd", "python -m pytest -q tests/test_a.py", "1 passed in 0.1s", 0)])]


def cx_looked():
    return [("look", [("cmd", "ls", "a.py", 0), ("cmd", "cat a.py", "x", 0)])]


def rollout(codex_home, cwd, sid, prompts, age_s=3600, source="cli", originator="codex-tui",
            archived=False, compressed=False, aborted=(), cut=False, meta_id=None):
    """A finished Codex session's rollout where Codex keeps it, last written `age_s` ago: per
    prompt one turn of commands (`bash -lc SCRIPT`) and file changes, ended by `task_complete`,
    or by `turn_aborted` for the indices in `aborted`; with `cut`, the last turn has no end."""
    end = time.time() - age_s
    start = end - 300
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime(start))
    d = (os.path.join(str(codex_home), "archived_sessions") if archived else
         os.path.join(str(codex_home), "sessions", *stamp[:10].split("-")))
    os.makedirs(d, exist_ok=True)
    recs = [{"timestamp": _iso(start), "type": "session_meta", "payload": {
        "id": meta_id or sid, "timestamp": _iso(start), "cwd": cwd, "originator": originator,
        "cli_version": "0.166.0", "source": source}}]
    n = 0
    for k, (text, calls) in enumerate(prompts):
        turn = f"turn-{k}"

        def ev(payload):
            recs.append({"type": "event_msg", "payload": dict(payload, turn_id=turn)})
        recs.append({"type": "turn_context", "payload": {"turn_id": turn, "cwd": cwd,
                                                         "model": "gpt-6.1-sol"}})
        ev({"type": "task_started"})
        ev({"type": "item_completed", "item": {"type": "UserMessage", "id": f"u{k}",
                                               "content": [{"type": "text", "text": text}]}})
        for c in calls:
            n += 1
            if c[0] == "cmd":
                item = {"type": "CommandExecution", "id": f"c{n}",
                        "command": ["bash", "-lc", c[1]], "cwd": cwd,
                        "aggregated_output": c[2], "exit_code": c[3],
                        "status": "completed" if c[3] == 0 else "failed"}
            else:
                item = {"type": "FileChange", "id": f"f{n}", "status": "completed",
                        "changes": {os.path.join(cwd, p): {"type": "update"} for p in c[1]}}
            ev({"type": "item_completed", "item": item})
        if k in aborted:
            ev({"type": "turn_aborted", "reason": "interrupted"})
        elif not (cut and k == len(prompts) - 1):
            ev({"type": "item_completed", "item": {"type": "AgentMessage", "id": f"a{k}",
                                                   "content": [{"type": "Text",
                                                                "text": "Done."}]}})
            ev({"type": "task_complete", "last_agent_message": "Done."})
        ev({"type": "token_count", "info": {"total_token_usage": {
            "input_tokens": 3000 * (k + 1), "cached_input_tokens": 2000 * (k + 1),
            "output_tokens": 90, "reasoning_output_tokens": 10}}})
    data = ("\n".join(json.dumps(r) for r in recs) + "\n").encode("utf-8")
    path = os.path.join(d, f"rollout-{stamp}-{sid}.jsonl")
    if compressed:
        path += ".zst"
        data = _zstd.compress(data) if _zstd else b"\x28\xb5\x2f\xfd" + data
    with open(path, "wb") as fh:
        fh.write(data)
    os.utime(path, (end, end))
    return path


def cx_ctx(env, repo):
    return LiveContext.build(hosts.Codex(), repo, env)


def cx_start(env, repo, sid, tmp_path):
    t = str(tmp_path / f"rollout-2026-01-01T00-00-00-{sid}.jsonl")
    open(t, "w").close()
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid,
                      "source": "startup", "transcript_path": t, "cwd": repo}, "codex")


def cx_folded_live(env, repo, tmp_path, n, prompts=1):
    """A Codex session of `prompts` prompts the plugin's own hooks saw end, folded."""
    t = rollout(tmp_path / "elsewhere", repo, cx(900 + n), cx_trap(n) * prompts)
    base = {"session_id": cx(900 + n), "transcript_path": t, "cwd": repo}
    call(env, dict(base, hook_event_name="SessionStart", source="startup"), "codex")
    call(env, dict(base, hook_event_name="SessionEnd", reason="other"), "codex")


def test_codex_history_is_read_by_the_rollouts_directory_oldest_first(env, repo, codex_home,
                                                                       tmp_path):
    rollout(codex_home, repo, cx(0), cx_trap(0), age_s=3600 * 10)
    rollout(codex_home, repo, cx(1), cx_trap(1), age_s=3600 * 9, archived=True)
    rollout(codex_home, repo, cx(2), cx_trap(2), age_s=3600 * 8, compressed=_zstd is not None)
    other = tmp_path / "other"
    other.mkdir()
    rollout(codex_home, str(other), cx(10), cx_trap(10))                    # another repository
    rollout(codex_home, str(tmp_path / "gone"), cx(11), cx_trap(11))        # its directory gone
    rollout(codex_home, repo, cx(12), cx_trap(12), age_s=5)                 # still writing
    rollout(codex_home, repo, cx(13), cx_trap(13), source="exec", originator="codex_exec")
    rollout(codex_home, repo, cx(14), cx_trap(14) * 2, source="exec", originator="Codex Desktop")
    rollout(codex_home, repo, cx(15), cx_trap(15),
            source={"subagent": {"thread_spawn": {"parent_thread_id": cx(0), "depth": 1}}})
    rollout(codex_home, repo, cx(16), [])                                   # no prompt
    rollout(codex_home, repo, cx(17), cx_trap(17), meta_id=cx(99))          # names another id

    first = cx_start(env, repo, cx(50), tmp_path)
    st = store_for(env, repo, "codex")
    assert ledger_ids(st) == [cx(0), cx(1), cx(2)]
    recs = st.read_lines("ledger.jsonl")
    assert all(r["imported"] and r["host"] == "codex" and r["untool_applied"] == []
               and r["prompts"] == 1 and "error" not in r for r in recs)
    assert all(e["imported"] for e in st.read("folded.json").values())
    doc = st.read("history.json")
    assert doc["state"] == "done" and doc["imported"] == 3
    assert doc["skipped"] == {"recent": 1, "headless": 2, "subagent": 1, "no_prompt": 1,
                              "unsafe_id": 1}
    assert "systemMessage" not in first
    second = cx_start(env, repo, cx(51), tmp_path)
    assert ("TokenEconomizer read 3 earlier session(s) of acme/widget from Codex's "
            "history.") in second["systemMessage"]
    assert not store_for(env, repo).read("folded.json")           # Claude Code's store: untouched


def test_codex_import_seeds_the_envelope_from_turns_that_ended(env, repo, codex_home):
    rollout(codex_home, repo, cx(0), cx_trap(0) + cx_looked(), age_s=7200)
    one = [("one", [("cmd", "git status", "clean", 0)])]
    rollout(codex_home, repo, cx(1), one + cx_trap(1), age_s=5400, aborted={0})
    rollout(codex_home, repo, cx(2), cx_trap(2) + cx_looked(), age_s=3600, cut=True)
    rollout(codex_home, repo, cx(3), [("fix", [
        ("cmd", "grep -rn raise src", 'src/a.py:1: raise RuntimeError("x")', 0),
        ("cmd", "python -m pytest -q", "FAILED tests/test_a.py::test_x - AssertionError", 1),
        ("cmd", "python -m pytest -q", "1 passed in 0.1s", 0)])], age_s=1800)
    ctx = cx_ctx(env, repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True)["imported"] == 4
    env_doc = ctx.store.read("envelope.json")
    # cx0: both turns; cx1: its first turn was stopped; cx2: its last turn never ended.
    assert env_doc["resolved_turns"] == [3, 2, 3, 3, 3]
    # The exit codes Codex recorded decide: an error-looking output of a command that exited
    # 0 is no failure, the test run that exited 1 is (without them, both would count).
    assert [s["error_outputs"] for s in env_doc["resolved_snaps"]] == [0, 0, 0, 0, 1]
    assert not os.path.exists(ctx.store.path("events.jsonl"))       # nothing was redirected


def test_te_history_import_and_status_on_codex(env, repo, codex_home, monkeypatch, capsys):
    for i in range(2):
        rollout(codex_home, repo, cx(i), cx_trap(i), age_s=3600 * (10 - i))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "codex", "history", "import", "--dry-run"]) == 0
    assert "2 earlier session(s) of acme/widget would be read" in capsys.readouterr().out
    assert store_for(env, repo, "codex").read("history.json") is None
    assert cli.main(["--cwd", repo, "--host", "codex", "history", "import"]) == 0
    assert "imported 2 earlier session(s) of acme/widget" in capsys.readouterr().out
    cli.main(["--cwd", repo, "--host", "codex", "status"])
    out = capsys.readouterr().out
    assert "sessions     0 folded (+2 from earlier history)" in out
    assert "read 2 earlier session(s) from Codex's history" in out
    cli.main(["--cwd", repo, "--host", "codex", "status", "--json"])
    js = json.loads(capsys.readouterr().out)
    assert js["ledger"]["imported_sessions"] == 2 and js["history"]["imported"] == 2
    assert js["toolprune"]["imported_sessions"] == 2 and js["toolprune"]["sessions"] == 0


def test_a_codex_store_with_folded_sessions_is_not_imported_into(env, repo, codex_home,
                                                                  tmp_path, monkeypatch, capsys):
    cx_folded_live(dict(env, TE_HISTORY_IMPORT="off"), repo, tmp_path, 0)
    rollout(codex_home, repo, cx(0), cx_trap(0))
    st = store_for(env, repo, "codex")
    assert list(st.read("folded.json")) == [cx(900)]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "codex", "history", "import"]) == 1
    assert capsys.readouterr().out.startswith(
        "not imported: acme/widget already has folded sessions")
    cx_start(env, repo, cx(50), tmp_path)
    doc = st.read("history.json")
    assert doc["state"] == "skipped" and "already" in doc["reason"]
    assert list(st.read("folded.json")) == [cx(900)]


def test_a_planned_rollout_is_found_after_codex_archived_or_compressed_it(env, repo, codex_home,
                                                                          monkeypatch):
    paths = [rollout(codex_home, repo, cx(i), cx_trap(i), age_s=3600 * (10 - i))
             for i in range(3)]
    ctx = cx_ctx(env, repo)
    clock = iter(range(1000))
    monkeypatch.setattr(history, "_clock", lambda: next(clock))
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True, budget_s=1.5)["state"] == "partial"
    # Meanwhile the person archives the second conversation, and Codex compresses the third.
    os.makedirs(os.path.join(str(codex_home), "archived_sessions"))
    os.replace(paths[1], os.path.join(str(codex_home), "archived_sessions",
                                      os.path.basename(paths[1])))
    if _zstd is not None:
        with open(paths[2], "rb") as fh:
            packed = _zstd.compress(fh.read())
        with open(paths[2] + ".zst", "wb") as fh:
            fh.write(packed)
        os.remove(paths[2])
    monkeypatch.setattr(history, "_clock", time.monotonic)
    res = history.run(ctx, auto=True)
    assert res["state"] == "done" and res["imported"] == 3 and res["skipped"] == {}
    assert ledger_ids(ctx.store) == [cx(0), cx(1), cx(2)]


@needs_zstd
def test_a_compressed_rollout_is_measured_by_what_it_decompresses_to(env, repo, codex_home,
                                                                     monkeypatch):
    noisy = [("build", [("cmd", "make", "warning: x\n" * 20000, 0)])]
    big = rollout(codex_home, repo, cx(0), noisy, compressed=True)
    rollout(codex_home, repo, cx(1), cx_trap(1), compressed=True)
    monkeypatch.setattr(history, "MAX_BYTES", 64 * 1024)
    assert os.path.getsize(big) < history.MAX_BYTES                 # small on disk
    p = history.plan(cx_ctx(env, repo))
    assert [s["id"] for s in p["sessions"]] == [cx(1)] and p["skipped"] == {"too_large": 1}


def test_a_compressed_rollout_is_left_out_where_python_cannot_read_it(env, repo, codex_home,
                                                                      monkeypatch):
    from tokeneconomizer.trace import jsonl
    rollout(codex_home, repo, cx(0), cx_trap(0), compressed=True)
    rollout(codex_home, repo, cx(1), cx_trap(1))
    monkeypatch.setattr(jsonl, "_zstd", None)
    p = history.plan(cx_ctx(env, repo))
    assert [s["id"] for s in p["sessions"]] == [cx(1)] and p["skipped"] == {"compressed": 1}


def test_an_imported_codex_session_holds_no_capability_under_auto(env, repo, codex_home,
                                                                  tmp_path):
    """Capabilities that went unused in Codex's history are proposed, but `auto` writes them
    only once the plugin's own sessions are enough evidence alone."""
    env = dict(env, TE_TOOLPRUNE="auto")
    for i in range(5):
        rollout(codex_home, repo, cx(i), cx_trap(i) + cx_looked(), age_s=3600 * (10 - i))
    cx_start(env, repo, cx(50), tmp_path)
    st = store_for(env, repo, "codex")
    unt = st.read("untool.json")
    assert st.read("history.json")["imported"] == 5
    assert unt["proposal"] and sorted(unt["needs_apply"]) == sorted(unt["proposal"])
    assert "auto_pending" not in unt and unt["imported_sessions"] == 5
    cx_folded_live(env, repo, tmp_path, 1)               # one live session: not enough alone
    unt = st.read("untool.json")
    assert "auto_pending" not in unt and sorted(unt["needs_apply"]) == sorted(unt["proposal"])
    assert not os.path.exists(denylist.settings_path(repo, "codex"))


def test_auto_holds_what_the_live_sessions_alone_would_have_held(env, repo, codex_home,
                                                                  tmp_path):
    """The same live sessions, with and without imported history before them: after each fold,
    `auto` holds the same capabilities. History must not hasten a demotion -- image tools need
    three unused sessions, and the controller demotes at most two capabilities per session."""
    env = dict(env, TE_TOOLPRUNE="auto", TE_TOOLPRUNE_QUIET_MINUTES="600")
    alone = dict(env, TOKENECONOMIZER_HOME=env["TOKENECONOMIZER_HOME"] + "-alone",
                 TE_HISTORY_IMPORT="off")
    for i in range(5):
        rollout(codex_home, repo, cx(i), cx_trap(i) * 2, age_s=3600 * (10 - i))
    cx_start(env, repo, cx(50), tmp_path)
    st, st_alone = store_for(env, repo, "codex"), store_for(alone, repo, "codex")
    assert st.read("history.json")["imported"] == 5

    def held(store):
        return set(((store.read("untool.json") or {}).get("auto_pending") or {}).get("items")
                   or ())
    ever = set()
    for n in range(1, 6):
        cx_folded_live(env, repo, tmp_path, n, prompts=15)
        cx_folded_live(alone, repo, tmp_path, n, prompts=15)
        assert held(st) == held(st_alone), n
        ever |= held(st)
        if n <= 3:
            assert "image_media" not in held(st), n
    assert "image_media" in ever                         # the live sessions do get there
    assert "image_media" in st.read("untool.json")["proposal"]
    assert not os.path.exists(denylist.settings_path(repo, "codex"))


def test_an_imported_codex_session_gives_back_no_rule_and_a_live_one_does(env, repo, codex_home,
                                                                         tmp_path, monkeypatch):
    monkeypatch.setattr(denylist, "codex_config_error", lambda root, **kw: None)
    env = dict(env, TE_TOOLPRUNE_PROTECT="image_media")
    ctx = cx_ctx(env, repo)
    assert sorted(denylist.apply(repo, ["image_media", "goals"], ctx.store, host="codex")) == [
        "goals", "image_media"]
    rollout(codex_home, repo, cx(0), cx_trap(0))
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    assert history.run(ctx, auto=True)["imported"] == 1
    assert sorted(denylist.applied_for(ctx.store, repo, "codex")) == ["goals", "image_media"]
    cx_folded_live(env, repo, tmp_path, 1)
    assert denylist.applied_for(ctx.store, repo, "codex") == ["goals"]


def test_where_codex_keeps_its_rollouts(tmp_path):
    home = tmp_path / "codex"
    day = home / "sessions" / "2026" / "11" / "02"
    day.mkdir(parents=True)
    (home / "archived_sessions").mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    a = f"rollout-2026-11-02T10-00-00-{cx(1)}.jsonl"
    for p in (day / a, day / (a + ".zst"), day / "notes.jsonl",
              home / "archived_sessions" / f"rollout-2026-11-01T09-00-00-{cx(2)}.jsonl.zst",
              extra / f"rollout-2026-10-01T09-00-00-{cx(3)}.jsonl"):
        p.write_text("{}\n")
    logs = hosts.Codex().history({"CODEX_HOME": str(home)}, [str(extra), str(home)])
    assert [(x["session_id"], os.path.basename(x["path"])) for x in logs] == [
        (cx(1), a), (cx(2), f"rollout-2026-11-01T09-00-00-{cx(2)}.jsonl.zst"),
        (cx(3), f"rollout-2026-10-01T09-00-00-{cx(3)}.jsonl")]
    assert trace.codex_rollout_id(f"rollout-2026-11-02T10-00-00-{cx(4)}.jsonl.zst") == cx(4)
    assert trace.codex_rollout_id("rollout-x.jsonl") is None
    assert trace.codex_home({"CODEX_HOME": "/c"}) == "/c"


def test_codex_turn_calls_are_what_the_live_hooks_saw(tmp_path):
    path = rollout(tmp_path / "codex", "/w", cx(0), cx_trap(0) + cx_looked(), aborted={1})
    turns = list(trace.codex_prompt_turns(path))
    assert len(turns) == 2
    assert trace.codex_turn_calls(turns[0]) == [
        ("apply_patch", {"path": os.path.join("/w", "pkg/mod0.py"),
                         "paths": [os.path.join("/w", "pkg/mod0.py")]}, "", None),
        ("bash", {"command": "pytest -q tests/test_a.py"}, "bash: pytest: command not found",
         127),
        ("bash", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s", 0)]
    assert trace.codex_turn_completed(turns[0]) and not trace.codex_turn_interrupted(turns[0])
    assert trace.codex_turn_interrupted(turns[1]) and not trace.codex_turn_completed(turns[1])
    head = trace.codex_rollout_head(path)
    assert head["session_id"] == cx(0) and head["cwd"] == "/w" and not head["headless"]
    exe = rollout(tmp_path / "codex", "/w", cx(1), cx_trap(1), source="exec",
                  originator="codex_exec", compressed=_zstd is not None)
    assert hosts.Codex().headless({}, exe) is (_zstd is not None)
