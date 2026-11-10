"""Sessions nobody watches (`codex exec`, `claude -p`): under `scripted_sessions=skip` (the
default) they are neither judged, redirected nor learned from, and their cost is counted apart.
"""
import json
import os

import pytest

from test_live_hooks import _guard_turn, _stuck, call, store_for, trap_session  # noqa: F401
from tokeneconomizer.plugin import accounting, cli, judge

GREEN = [("Read", {"file_path": "/r/a.py"}, "ok")] * 8


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    return str(r)


@pytest.fixture
def judged(monkeypatch):
    jobs = []
    monkeypatch.setattr(judge, "spawn_job", lambda ctx, sid, job: jobs.append(job))
    return jobs


def exec_rollout(path, repo, source="exec", originator="codex_exec"):
    """The head of a Codex rollout, as `codex exec` writes it, with its token count."""
    recs = [{"type": "session_meta", "payload": {"id": os.path.basename(path), "cwd": repo,
                                                 "source": source, "originator": originator}},
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 3000, "cached_input_tokens": 2000,
                                      "output_tokens": 90}}}}]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
    return path


def warm(env, repo, host):
    """Five ended-green turns of eight calls in a watched session: the guard has an envelope,
    and a 40-call stuck turn is past its backstop."""
    for k in range(5):
        _guard_turn(env, repo, "w", f"g{k}", GREEN, host)
    return store_for(env, repo, host).read("envelope.json")["resolved_turns"]


def stuck_session(env, repo, sid, transcript, host, calls=None):
    """A whole session: start, one turn (40 stuck calls, or `calls`), end. Returns the turn's
    outputs."""
    idk = "turn_id" if host == "codex" else "prompt_id"
    base = {"session_id": sid, "transcript_path": transcript, "cwd": repo}
    call(env, dict(base, hook_event_name="SessionStart", source="startup"), host)
    call(env, dict(base, **{idk: "t1"}, hook_event_name="UserPromptSubmit", prompt="go"), host)
    from test_live_hooks import _post
    outs = [call(env, _post(dict(base, **{idk: "t1"}), n, i, r), host)
            for n, i, r in (calls or _stuck(host))]
    call(env, dict(base, **{idk: "t1"}, hook_event_name="Stop", stop_hook_active=False), host)
    call(env, dict(base, hook_event_name="SessionEnd", reason="other"), host)
    return outs


def record(env, repo, sid, host):
    return next(r for r in store_for(env, repo, host).read_lines("ledger.jsonl")
                if r.get("kind") == "session" and r["session_id"] == sid)


def test_a_codex_exec_run_is_neither_judged_nor_learned_from(env, repo, tmp_path, judged):
    envelope = warm(env, repo, "codex")
    st = store_for(env, repo, "codex")
    untool_before = st.read("untool.json")
    outs = stuck_session(env, repo, "x1", exec_rollout(str(tmp_path / "x1.jsonl"), repo),
                         "codex")
    assert all(o == {} for o in outs) and judged == []         # no redirect, no paid judge
    rec = record(env, repo, "x1", "codex")
    assert rec["scripted"] and rec["effective_tokens"] and rec["injection"] is None
    assert st.read("untool.json") == untool_before
    assert st.read("folded.json")["x1"]["scripted"] is True


def test_with_learn_a_codex_exec_run_is_guarded_and_learned_from(env, repo, tmp_path, judged):
    env = dict(env, TE_SCRIPTED_SESSIONS="learn")
    envelope = warm(env, repo, "codex")
    outs = stuck_session(env, repo, "x1", exec_rollout(str(tmp_path / "x1.jsonl"), repo),
                         "codex")
    assert judged or any("hookSpecificOutput" in o for o in outs)
    rec = record(env, repo, "x1", "codex")
    assert "scripted" not in rec and rec["injection"] is not None


@pytest.mark.parametrize("setting,learned", [("skip", False), ("learn", True)])
def test_a_scripted_turn_that_ends_green_feeds_the_envelope_only_with_learn(
        env, repo, tmp_path, setting, learned):
    env = dict(env, TE_SCRIPTED_SESSIONS=setting)
    envelope = warm(env, repo, "codex")
    stuck_session(env, repo, "x1", exec_rollout(str(tmp_path / "x1.jsonl"), repo), "codex",
                  calls=GREEN[:6])
    after = store_for(env, repo, "codex").read("envelope.json")["resolved_turns"]
    assert after == (envelope + [6] if learned else envelope)


def test_a_watched_codex_session_is_learned_from(env, repo, tmp_path, judged):
    warm(env, repo, "codex")
    stuck_session(env, repo, "c1", exec_rollout(str(tmp_path / "c1.jsonl"), repo, source="cli",
                                                originator="codex-tui"), "codex")
    assert "scripted" not in record(env, repo, "c1", "codex")


@pytest.mark.parametrize("attended,entrypoint,scripted", [
    ("0", "sdk-cli", True), ("1", "claude-desktop", False), ("", "sdk-ts", True),
    ("", "cli", False)])
def test_claude_code_tells_a_scripted_run_by_its_environment(env, repo, tmp_path, judged,
                                                             attended, entrypoint, scripted):
    warm(env, repo, "claude")
    run_env = dict(env, CLAUDE_CODE_ENTRYPOINT=entrypoint,
                   **({"CLAUDE_CODE_SESSION_ATTENDED": attended} if attended else {}))
    outs = stuck_session(run_env, repo, "p1", trap_session(str(tmp_path / "p1.jsonl"), repo, 1),
                         "claude")
    rec = record(env, repo, "p1", "claude")
    assert bool(rec.get("scripted")) is scripted
    if scripted:
        assert all(o == {} for o in outs) and judged == []
    else:
        assert judged or any("hookSpecificOutput" in o for o in outs)


def test_mode_interactive_overrides_what_the_log_says(env, repo, tmp_path, judged):
    env = dict(env, TE_MODE="interactive")
    warm(env, repo, "codex")
    outs = stuck_session(env, repo, "x1", exec_rollout(str(tmp_path / "x1.jsonl"), repo),
                         "codex")
    assert judged or any("hookSpecificOutput" in o for o in outs)     # guarded as watched
    assert "scripted" not in record(env, repo, "x1", "codex")


def test_scripted_sessions_are_counted_apart(env, repo, tmp_path, monkeypatch, capsys):
    s = accounting.summarize([
        {"kind": "session", "effective_tokens": 100.0, "requests": 2},
        {"kind": "session", "effective_tokens": 900.0, "requests": 9, "scripted": True}])
    assert s["sessions"] == 1 and s["scripted_sessions"] == 1 and s["effective_tokens"] == 100.0
    stuck_session(env, repo, "x1", exec_rollout(str(tmp_path / "x1.jsonl"), repo), "codex")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cli.main(["--cwd", repo, "--host", "codex", "status"])
    assert "sessions     0 folded (+1 scripted, not learned from)" in capsys.readouterr().out


def test_a_conversation_a_script_resumes_becomes_scripted(env, repo, tmp_path, judged):
    warm(env, repo, "claude")
    t = trap_session(str(tmp_path / "r1.jsonl"), repo, 1)
    base = {"session_id": "r1", "transcript_path": t, "cwd": repo}
    call(dict(env, CLAUDE_CODE_SESSION_ATTENDED="1"),
         dict(base, hook_event_name="SessionStart", source="startup"), "claude")
    outs = stuck_session(dict(env, CLAUDE_CODE_SESSION_ATTENDED="0"), repo, "r1", t, "claude")
    assert all(o == {} for o in outs) and judged == []
    assert record(env, repo, "r1", "claude")["scripted"] is True
