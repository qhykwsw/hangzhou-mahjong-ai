"""Transactional ownership of research state and idempotent job receipts."""

from contextlib import contextmanager
import json
import sqlite3
import time
from mahjong_ai.project import ProjectDescriptor


class Store:
    def __init__(self, project):
        self.project = ProjectDescriptor.load(project)
        self.path = self.project.service / "research.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        with self.connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS registry(id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, experiment TEXT, at REAL, kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS receipts(experiment TEXT, job TEXT, payload TEXT, PRIMARY KEY(experiment,job));""")

        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        # Every edit rewrites the whole state blob, which takes seconds once a
        # long experiment has accumulated its job records. Writers queue behind
        # each other; a short wait turned that queue into a paused experiment.
        db = sqlite3.connect(self.path, timeout=300)
        db.execute("PRAGMA busy_timeout=300000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, state):
        with self.connect() as db:
            db.execute("INSERT INTO registry VALUES(1,?)", (json.dumps(state),))

    def read(self, experiment=None):
        with self.connect() as db:
            row = db.execute("SELECT state FROM registry WHERE id=1").fetchone()
        if not row:
            return None
        state = json.loads(row[0])
        if experiment is not None and state["id"] != experiment:
            raise ValueError("该实验已不是当前实验")
        return state

    @contextmanager
    def edit(self, experiment=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM registry WHERE id=1").fetchone()
            if not row:
                raise ValueError("策略注册表尚未初始化")
            state = json.loads(row[0])
            if experiment is not None and state["id"] != experiment:
                raise ValueError("该实验已不是当前实验")
            yield state, db
            db.execute("UPDATE registry SET state=? WHERE id=1", (json.dumps(state),))

    def event(self, db, state, kind, payload):
        db.execute(
            "INSERT INTO events(experiment,at,kind,payload) VALUES(?,?,?,?)",
            (state["id"], time.time(), kind, json.dumps(payload, ensure_ascii=False)),
        )

    def receipt(self, db, experiment, job, payload, attempt=0):
        """Commit one immutable result for one execution attempt.

        A retried job keeps its durable job id, but each retry is a new
        execution attempt.  The first attempt deliberately keeps the legacy
        key so existing databases remain readable; later attempts use a
        namespaced key and cannot collide with an earlier result.
        """
        receipt_job = job if not attempt else f"{job}#attempt-{attempt}"
        encoded = json.dumps(payload, sort_keys=True)
        old = db.execute(
            "SELECT payload FROM receipts WHERE experiment=? AND job=?",
            (experiment, receipt_job),
        ).fetchone()
        if old and old[0] != encoded:
            raise ValueError("不可变回执发生变化")
        db.execute(
            "INSERT OR IGNORE INTO receipts VALUES(?,?,?)",
            (experiment, receipt_job, encoded),
        )

    def events(self, experiment, after=0):
        with self.connect() as db:
            return [
                dict(seq=r[0], at=r[1], kind=r[2], data=json.loads(r[3]))
                for r in db.execute(
                    "SELECT seq,at,kind,payload FROM events WHERE experiment=? AND seq>? ORDER BY seq DESC LIMIT 200",
                    (experiment, after),
                )
            ][::-1]
