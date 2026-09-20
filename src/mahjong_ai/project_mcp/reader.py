"""Allowlisted, bounded project readers; no caller-supplied command execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path, PurePosixPath

MAX_FILE = 256 * 1024
MAX_RESULT = 96 * 1024
EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".ini", ".sql", ".html", ".css", ".js", ".ts", ".sh", ".c", ".h", ".xml"}
ROOT_FILES = {"README.md", "AGENTS.md", "CLAUDE.md", "CONTEXT.md", "pyproject.toml", "setup.py"}
EXCLUDED = {"archive", "archives", "runs", "models", "releases", "build", "dist", "logs", "state", "node_modules", "__pycache__", "credentials", "secrets"}
SENSITIVE_KEY = re.compile(r"(?i)(?:password|passwd|secret|credential|authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|private[_-]?key|(?:^|[._-])token(?:$|[._-]))")
ASSIGNMENT = re.compile(r'''(?im)(?<![\w.-])(["']?[\w.-]{0,80}(?:password|passwd|secret|credential|authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|\btoken)[\w.-]{0,80}["']?[ \t]{0,20}[:=][ \t]{0,20})[^\r\n]+''')
TOKEN = re.compile(r"(?i)\b(?:Bearer\s+[^\s\"'<>]+|sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{15,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)")
PEM = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL)
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,250}\.[A-Za-z]{2,24}\b")
URL_SECRET = re.compile(r"(?i)([?&](?:token|key|code|signature|sig|password)=)[^\s&#\"']+")
URL_USERINFO = re.compile(r"(https?://)[^\s/@]{1,200}:[^\s/@]{1,200}@")


def redact(text: str) -> str:
    text = PEM.sub(lambda m: "[REDACTED PRIVATE KEY]" + "\n" * m[0].count("\n"), text)
    text = TOKEN.sub("[REDACTED]", text)
    text = ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = URL_SECRET.sub(r"\1[REDACTED]", text)
    text = URL_USERINFO.sub(r"\1[REDACTED]@", text)
    return EMAIL.sub("[REDACTED EMAIL]", text)


def sanitize(value):
    if isinstance(value, dict):
        return {redact(str(k)): "[REDACTED]" if SENSITIVE_KEY.search(str(k)) else sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return redact(value) if isinstance(value, str) else value


def bounded_result(value: dict) -> dict:
    result = sanitize(value)
    if len(json.dumps(result, ensure_ascii=False, default=str).encode()) > MAX_RESULT:
        raise ValueError("Result too large; narrow the query or use a smaller page")
    return result


def allowed(path: str) -> bool:
    parts = path.split("/")
    if not path or "\\" in path or "\x00" in path:
        return False
    if any(not p or p in {".", ".."} or p.startswith(".") or p in EXCLUDED or p.endswith("_bak") for p in parts):
        return False
    if SENSITIVE_KEY.search(parts[-1]) or parts[-1].lower().endswith((".pem", ".key", ".env")):
        return False
    in_scope = path in ROOT_FILES or parts[0] in {"src", "scripts", "projects", "docs", "tests"}
    return in_scope and (path in ROOT_FILES or PurePosixPath(path).suffix in EXTENSIONS)


def git(root: Path, *arguments: str, limit: int = 2 * 1024 * 1024) -> bytes:
    """Bound output while the process runs, and kill on a wall-clock deadline."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    with subprocess.Popen(
        ["git", "--no-pager", "--literal-pathspecs", "-C", str(root), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
    ) as process:
        result = bytearray()
        deadline = time.monotonic() + 3
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise ValueError("Git read timed out; narrow the query")
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    result.extend(chunk)
                    if len(result) > limit:
                        raise ValueError("Git result too large; narrow the query")
            if process.wait(timeout=max(0.01, deadline - time.monotonic())):
                raise ValueError("Git source or revision is unavailable")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return bytes(result)


class RepositoryReader:
    def __init__(self, root: Path):
        self.root, self.source = root.resolve(strict=True), "mahjong"

    def revision(self, revision: str) -> str:
        if revision != "HEAD" and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be HEAD or a full commit hash")
        return git(self.root, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()

    def paths(self, revision: str | None = None) -> list[str]:
        raw = git(self.root, "ls-tree", "-r", "--name-only", "-z", revision) if revision else git(self.root, "ls-files", "-z")
        return sorted({p for p in raw.decode().split("\x00") if allowed(p)})

    def content(self, path: str, revision: str | None = None) -> str:
        if not allowed(path):
            raise ValueError("Path is outside the readable scope")
        if revision:
            # Reject historical symlinks too: a symlink blob is not source text.
            info = git(self.root, "ls-tree", revision, "--", path).decode()
            if not info.startswith("100"):
                raise ValueError("Only regular tracked files are readable")
            raw = git(self.root, "show", f"{revision}:{path}", limit=MAX_FILE)
        else:
            import stat
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parts = path.split("/")
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    os.close(directory)
                    directory = child
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            finally:
                os.close(directory)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Only regular files are readable")
                raw = stream.read(MAX_FILE + 1)
            if len(raw) > MAX_FILE:
                raise ValueError("File exceeds the read limit")
        if b"\x00" in raw:
            raise ValueError("Binary files are outside the readable scope")
        return redact(raw.decode("utf-8"))

    def read(self, path: str, start: int, lines: int, revision: str | None) -> dict:
        commit = self.revision(revision) if revision else None
        if path not in self.paths(commit):
            raise ValueError("Choose a tracked path returned by list_files")
        body = self.content(path, commit)
        all_lines = body.splitlines()
        selected = all_lines[start - 1:start - 1 + lines]
        text = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start))
        return {"source": self.source, "path": path, "revision": commit or "working_tree",
                "redacted_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "start_line": start, "total_lines": len(all_lines), "text": text,
                "next_line": start + len(selected) if start + len(selected) <= len(all_lines) else None}

    def search(self, query: str, prefix: str, offset: int) -> dict:
        paths = [p for p in self.paths() if p.startswith(prefix)]
        results, scanned, skipped = [], 0, 0
        deadline, size = time.monotonic() + 2, 0
        for path in paths[offset:]:
            if scanned >= 100 or time.monotonic() > deadline or size >= 4 * 1024 * 1024 or len(results) >= 40:
                break
            scanned += 1
            try:
                body = self.content(path)
            except (OSError, ValueError, UnicodeError):
                skipped += 1
                continue
            size += len(body.encode())
            for number, line in enumerate(body.splitlines(), 1):
                if query.casefold() in line.casefold():
                    results.append({"path": path, "line": number, "text": line[:500]})
                    break  # one hit/file; read_file gets the surrounding evidence
        return {"source": self.source, "results": results, "scanned_files": scanned,
                "skipped_files": skipped, "total_files": len(paths),
                "next_offset": offset + scanned if offset + scanned < len(paths) else None}

    def history(self, path: str, limit: int) -> dict:
        if path not in self.paths():
            raise ValueError("Choose a tracked path returned by list_files")
        body = git(self.root, "log", f"-{limit}", "--format=%H%x09%cI%x09%s", "--", path, limit=MAX_RESULT)
        return {"source": self.source, "path": path, "history": redact(body.decode()),
                "note": "Local Git history only; GitHub PR discussions are not included."}
