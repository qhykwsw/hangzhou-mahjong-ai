"""Transactional ownership of research state and idempotent job receipts."""

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time


class Store:
    def __init__(self, project):
        self.path = Path(project) / ".local/service/research.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        with self.connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run TEXT, at REAL, kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS receipts(run TEXT, job TEXT, payload TEXT, PRIMARY KEY(run,job));""")

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

    def create(self, state):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT state FROM runs"):
                if json.loads(row[0])["status"] in ("running", "pausing"):
                    raise ValueError("pause the active research first")
            db.execute("INSERT INTO runs VALUES(?,?)", (state["id"], json.dumps(state)))

    def read(self, run):
        with self.connect() as db:
            row = db.execute("SELECT state FROM runs WHERE id=?", (run,)).fetchone()
        if not row:
            raise ValueError("unknown research")
        return json.loads(row[0])

    def runs(self):
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT state FROM runs ORDER BY rowid DESC")
            ]

    @contextmanager
    def edit(self, run):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM runs WHERE id=?", (run,)).fetchone()
            if not row:
                raise ValueError("unknown research")
            state = json.loads(row[0])
            yield state, db
            db.execute("UPDATE runs SET state=? WHERE id=?", (json.dumps(state), run))

    def event(self, db, state, kind, payload):
        db.execute(
            "INSERT INTO events(run,at,kind,payload) VALUES(?,?,?,?)",
            (state["id"], time.time(), kind, json.dumps(payload, ensure_ascii=False)),
        )

    def receipt(self, db, run, job, payload):
        encoded = json.dumps(payload, sort_keys=True)
        old = db.execute(
            "SELECT payload FROM receipts WHERE run=? AND job=?", (run, job)
        ).fetchone()
        if old and old[0] != encoded:
            raise ValueError("immutable receipt changed")
        db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?,?)", (run, job, encoded))

    def events(self, run, after=0):
        with self.connect() as db:
            return [
                dict(seq=r[0], at=r[1], kind=r[2], data=json.loads(r[3]))
                for r in db.execute(
                    "SELECT seq,at,kind,payload FROM events WHERE run=? AND seq>? ORDER BY seq DESC LIMIT 200",
                    (run, after),
                )
            ][::-1]
