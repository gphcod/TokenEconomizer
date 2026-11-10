"""Where a deny list may be written: never through a folder that is a symlink, a junction or a
file, and never recorded when it was not written.

A repository can hold `.claude -> .agents` as a git symlink. Where symlinks work the write would
land in `.agents/`, which the plugin's exclude rule does not cover; on Windows without symlink
rights git checks the link out as a small file, which no folder can be made over.
"""
import os
import time

import pytest

from tokeneconomizer.plugin import cli, denylist, fold, hosts
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.store import Store


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "GIT_CEILING_DIRECTORIES": str(tmp_path),
            "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".agents" / "skills").mkdir(parents=True)
    return str(r)


def placeholder(repo, name=".claude", target=".agents"):
    """A git symlink as Git for Windows checks it out without symlink rights: a file."""
    with open(os.path.join(repo, name), "w", encoding="utf-8") as fh:
        fh.write(target)


def linked_folder(repo, name, target):
    """A folder link inside the repository: a junction on Windows (no rights needed), else a
    symlink."""
    src, dst = os.path.join(repo, target), os.path.join(repo, name)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def test_a_settings_folder_that_is_a_file_is_refused_and_nothing_is_recorded(repo, tmp_path):
    placeholder(repo)
    st = Store(str(tmp_path / "st"))
    with pytest.raises(RuntimeError, match="not a plain folder"):
        denylist.apply(repo, ["CronCreate", "LSP"], st, host="claude")
    assert denylist.applied_for(st, repo, "claude") == []
    assert os.path.isfile(os.path.join(repo, ".claude"))


@pytest.mark.parametrize("host,name", [("claude", ".claude"), ("copilot", ".github"),
                                       ("codex", ".codex")])
def test_a_linked_folder_inside_the_repository_is_refused(repo, tmp_path, host, name):
    """Copilot writes `.github/copilot/settings.local.json`: a link two folders up counts."""
    linked_folder(repo, name, ".agents")
    st = Store(str(tmp_path / "st"))
    item = {"claude": "CronCreate", "copilot": "github", "codex": "goals"}[host]
    with pytest.raises(RuntimeError, match="not a plain folder"):
        denylist.apply(repo, [item], st, host=host)
    assert denylist.applied_for(st, repo, host) == []
    assert os.listdir(os.path.join(repo, ".agents")) == ["skills"]


def test_a_write_that_fails_puts_the_record_back(repo, tmp_path, monkeypatch):
    st = Store(str(tmp_path / "st"))
    assert denylist.apply(repo, ["CronCreate"], st, host="claude") == ["CronCreate"]

    def full_disk(path, obj):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(denylist, "write_json", full_disk)
    with pytest.raises(OSError):
        denylist.apply(repo, ["LSP"], st, host="claude")
    assert denylist.applied_for(st, repo, "claude") == ["CronCreate"]


def test_auto_records_the_refusal_not_the_tools(repo, env):
    placeholder(repo)
    ctx = LiveContext.build(hosts.ClaudeCode(), repo, dict(env, TE_TOOLPRUNE="auto"))
    ctx.store.write("untool.json", {"proposal": ["CronCreate", "LSP"], "auto_pending": {
        "items": ["CronCreate", "LSP"], "root": repo, "since": time.time() - 10000}})
    assert fold.apply_pending(ctx) is None
    unt = ctx.store.read("untool.json")
    assert "not a plain folder" in unt["auto_apply_refused"] and "auto_pending" not in unt
    assert denylist.applied_for(ctx.store, repo, "claude") == []


def test_toolprune_apply_says_why_it_cannot_write(repo, env, monkeypatch, capsys):
    placeholder(repo)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--host", "claude", "--cwd", repo, "toolprune", "apply", "CronCreate"]) == 1
    assert "not applied: refusing to write" in capsys.readouterr().out


def full_disk(path, obj):
    raise OSError(28, "No space left on device")


def test_auto_reports_a_write_that_fails_and_records_nothing(repo, env, monkeypatch):
    ctx = LiveContext.build(hosts.ClaudeCode(), repo, dict(env, TE_TOOLPRUNE="auto"))
    ctx.store.write("untool.json", {"proposal": ["CronCreate"], "auto_pending": {
        "items": ["CronCreate"], "root": repo, "since": time.time() - 10000}})
    monkeypatch.setattr(denylist, "write_json", full_disk)
    assert fold.apply_pending(ctx) is None
    assert "No space left" in ctx.store.read("untool.json")["auto_apply_refused"]
    assert denylist.applied_for(ctx.store, repo, "claude") == []


def test_toolprune_apply_reports_a_write_that_fails(repo, env, monkeypatch, capsys):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(denylist, "write_json", full_disk)
    assert cli.main(["--host", "claude", "--cwd", repo, "toolprune", "apply", "CronCreate"]) == 1
    assert "not applied:" in capsys.readouterr().out


def test_a_junction_is_refused_without_isjunction(repo, tmp_path, monkeypatch):
    """`os.path.isjunction` arrived in Python 3.12; the guard must not need it."""
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    linked_folder(repo, ".claude", ".agents")
    with pytest.raises(RuntimeError, match="not a plain folder"):
        denylist.apply(repo, ["CronCreate"], Store(str(tmp_path / "st")), host="claude")


def test_one_writer_at_a_time(repo, tmp_path, monkeypatch):
    """The fold, the CLI and `apply-pending` change the settings file and its record together;
    while another holds the deny list, a write is refused and records nothing."""
    import threading
    st = Store(str(tmp_path / "st"))
    monkeypatch.setattr(denylist, "LOCK_WAIT_S", 0.2)
    holding, release = threading.Event(), threading.Event()

    def other_writer():
        with st.locked("denylist"):
            holding.set()
            release.wait(10)
    th = threading.Thread(target=other_writer)
    th.start()
    try:
        assert holding.wait(10)
        with pytest.raises(RuntimeError, match="under way"):
            denylist.apply(repo, ["CronCreate"], st, host="claude")
        with pytest.raises(RuntimeError, match="under way"):
            denylist.set_kept(st, repo, "claude", add=["LSP"])
        assert denylist.applied_for(st, repo, "claude") == []
        assert denylist.kept_for(st, repo, "claude") == []
    finally:
        release.set()
        th.join()
    assert denylist.apply(repo, ["CronCreate"], st, host="claude") == ["CronCreate"]


def test_a_caller_holds_the_lock_across_its_steps(repo, tmp_path, monkeypatch):
    """`toolprune apply` writes, then marks the names as no longer given back: one lock over
    both, taken again inside without waiting."""
    st = Store(str(tmp_path / "st"))
    monkeypatch.setattr(denylist, "LOCK_WAIT_S", 0.2)
    with denylist.one_writer(st):
        assert denylist.apply(repo, ["CronCreate"], st, host="claude") == ["CronCreate"]
        denylist.set_kept(st, repo, "claude", remove=["CronCreate"])
    assert denylist.applied_for(st, repo, "claude") == ["CronCreate"]
