"""Persistent registry for immutable cross-project competition entrants."""

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time


class CompetitionRegistry:
    """Store release metadata separately from an automatic research registry."""

    def __init__(self, root: str | Path):
        root = Path(root).resolve()
        self.path = root / ".local/competition/competition.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript(
                """PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS releases(
                    entrant_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    bundle_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resident INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS release_version
                    ON releases(project_id, version);
                CREATE TABLE IF NOT EXISTS competitions(
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    config TEXT NOT NULL,
                    entrants TEXT NOT NULL,
                    results TEXT,
                    created_at REAL NOT NULL
                );"""
            )
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, manifest: dict, bundle_path: str | Path, *, resident=False):
        required = ("entrant_id", "project_id", "version", "content_hash")
        if any(
            not isinstance(manifest.get(key), str) or not manifest[key]
            for key in required
        ):
            raise ValueError(
                "release manifest requires entrant_id, project_id, version and content_hash"
            )
        now = time.time()
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        with self.connect() as db:
            try:
                db.execute(
                    "INSERT INTO releases VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        manifest["entrant_id"],
                        manifest["project_id"],
                        manifest["version"],
                        manifest["content_hash"],
                        str(Path(bundle_path).resolve()),
                        payload,
                        "registered",
                        int(resident),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "release version or entrant already registered"
                ) from exc
        return self.get(manifest["entrant_id"])

    def get(self, entrant_id: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM releases WHERE entrant_id=?", (entrant_id,)
            ).fetchone()
        if not row:
            raise KeyError(entrant_id)
        return self._release(row)

    def list(self, *, active_only=False, resident_only=False):
        query = "SELECT * FROM releases"
        clauses, args = [], []
        if active_only:
            clauses.append("status='registered'")
        if resident_only:
            clauses.append("resident=1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, entrant_id"
        with self.connect() as db:
            return [self._release(row) for row in db.execute(query, args)]

    def set_status(self, entrant_id: str, status: str):
        if status not in {"registered", "paused", "revoked"}:
            raise ValueError("unsupported release status")
        with self.connect() as db:
            changed = db.execute(
                "UPDATE releases SET status=?,updated_at=? WHERE entrant_id=?",
                (status, time.time(), entrant_id),
            ).rowcount
        if not changed:
            raise KeyError(entrant_id)
        return self.get(entrant_id)

    def set_resident(self, entrant_id: str, resident: bool):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE releases SET resident=?,updated_at=? WHERE entrant_id=?",
                (int(resident), time.time(), entrant_id),
            ).rowcount
        if not changed:
            raise KeyError(entrant_id)
        return self.get(entrant_id)

    @staticmethod
    def _release(row):
        return {
            "entrant_id": row[0],
            "project_id": row[1],
            "version": row[2],
            "content_hash": row[3],
            "bundle_path": row[4],
            "metadata": json.loads(row[5]),
            "status": row[6],
            "resident": bool(row[7]),
            "created_at": row[8],
            "updated_at": row[9],
        }
