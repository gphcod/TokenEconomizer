"""Reading a JSONL log line by line, skipping what does not parse.

A Zstandard-compressed log (`*.zst`: Codex compresses its older rollouts) is read through the
standard library's `compression.zstd`, which Python has from 3.14; before that it cannot be read
(`readable`), and a reader treats it as missing.
"""
import json
import os

try:
    from compression import zstd as _zstd
    _READ_ERRORS = (OSError, EOFError, _zstd.ZstdError)
except ImportError:                     # Python < 3.14
    _zstd = None
    _READ_ERRORS = (OSError, EOFError)

_CHUNK = 1 << 20


class Unreadable(OSError):
    """A log that could not be read to its end: missing, compressed without the means to
    decompress it here, or corrupt."""


def compressed(path):
    return str(path or "").endswith(".zst")


def readable(path):
    """Can this Python read `path`'s format? Everything but a `.zst` log before 3.14."""
    return _zstd is not None or not compressed(path)


def open_text(path):
    """`path` opened for reading text (UTF-8, undecodable bytes replaced), decompressed where it
    is a `.zst` log. Raises OSError where it cannot be."""
    if compressed(path):
        if _zstd is None:
            raise Unreadable(f"{path}: Zstandard-compressed, and this Python has no zstd")
        return _zstd.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def decompressed_size(path, cap):
    """The bytes `path` holds once decompressed: its size on disk, or for a `.zst` log the bytes
    it decompresses to, counted only up to `cap` (more is returned as `cap + 1`). Raises OSError
    where it cannot be read."""
    if not compressed(path):
        return os.path.getsize(path)
    if _zstd is None:
        raise Unreadable(f"{path}: Zstandard-compressed, and this Python has no zstd")
    n = 0
    try:
        with _zstd.open(path, "rb") as fh:
            while n <= cap:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                n += len(chunk)
    except _READ_ERRORS as e:
        raise Unreadable(str(e)) from e
    return min(n, cap + 1)


def lines(path, strict=False):
    """Non-empty, stripped lines of a text file; nothing when it cannot be read. `strict`: raise
    `Unreadable` where it cannot be read to its end, rather than stop where reading stopped."""
    try:
        with open_text(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line
    except _READ_ERRORS as e:
        if strict:
            raise Unreadable(str(e)) from e
        return


def records(path):
    """Each line of a JSONL file that parses to a JSON object."""
    for line in lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            yield ev


def head_records(path, max_lines, max_bytes, line_bytes):
    """The JSON objects among a log's first `max_lines` lines, reading at most `max_bytes`; a line
    longer than `line_bytes` is skipped, not parsed. Stops quietly where the log cannot be read."""
    try:
        with open_text(path) as fh:
            left = max_bytes
            for _ in range(max_lines):
                line = fh.readline(min(left, line_bytes)) if left > 0 else ""
                if not line:
                    return
                left -= len(line)
                if not line.endswith("\n") and len(line) >= line_bytes:
                    while left > 0:
                        rest = fh.readline(min(left, line_bytes))
                        left -= len(rest)
                        if not rest or rest.endswith("\n"):
                            break
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    yield ev
    except _READ_ERRORS:
        return
