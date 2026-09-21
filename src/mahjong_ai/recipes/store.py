"""Durable state of recipe iteration, separate from the rule research registry."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import tomllib

# The recipe project, relative to the repository workspace.
PROJECT = "projects/personal/selfplay"


def configuration(project):
    """The frozen conditions a new experiment copies from recipes/iteration.toml."""
    config = tomllib.loads((Path(project) / "recipes/iteration.toml").read_text())
    agent, trial = config["agent"], config["trial"]
    for key in ("seconds", "continuations", "smoke_limit"):
        if type(agent[key]) is not int or agent[key] < 0:
            raise ValueError(f"agent.{key} must be a non-negative integer")
    for key in (
        "budget_seconds",
        "smoke_seconds",
        "evaluation_games",
        "brawl_appearances",
        "limit",
    ):
        if type(trial[key]) is not int or trial[key] < 1:
            raise ValueError(f"trial.{key} must be a positive integer")
    if trial["evaluation_games"] % 4 or trial["brawl_appearances"] % 16:
        raise ValueError("evaluation needs whole seat rotations and brawl whole blocks")
    ids = [line["id"] for line in config["lines"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("recipe lines need distinct ids")
    if len({line["host"] for line in config["lines"]}) != len(ids):
        raise ValueError("each recipe line needs its own host")
    return config


class RecipeStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.project = self.workspace / PROJECT
        self.root = self.project / ".local/recipes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.path = self.root / "recipes.sqlite"
        with self.connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, created REAL NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, experiment TEXT, at REAL, kind TEXT, payload TEXT);""")
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

    def current(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM experiments ORDER BY created DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def create(self, state):
        with self.connect() as db:
            db.execute(
                "INSERT INTO experiments VALUES(?,?,?)",
                (state["id"], state["created_at"], json.dumps(state)),
            )

    def read(self, experiment=None):
        experiment = experiment or self.current()
        if experiment is None:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT state FROM experiments WHERE id=?", (experiment,)
            ).fetchone()
        if not row:
            raise ValueError("未知的训练方案迭代实验")
        return json.loads(row[0])

    def experiments(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, created FROM experiments ORDER BY created DESC"
            ).fetchall()
        return [{"id": r[0], "created_at": r[1]} for r in rows]

    @contextmanager
    def edit(self, experiment=None):
        experiment = experiment or self.current()
        if experiment is None:
            raise ValueError("还没有训练方案迭代实验")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state FROM experiments WHERE id=?", (experiment,)
            ).fetchone()
            if not row:
                raise ValueError("未知的训练方案迭代实验")
            state = json.loads(row[0])
            yield state, db
            db.execute(
                "UPDATE experiments SET state=? WHERE id=?",
                (json.dumps(state), experiment),
            )

    def event(self, db, state, kind, payload):
        db.execute(
            "INSERT INTO events(experiment,at,kind,payload) VALUES(?,?,?,?)",
            (state["id"], time.time(), kind, json.dumps(payload, ensure_ascii=False)),
        )

    def events(self, experiment, limit=100):
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,at,kind,payload FROM events WHERE experiment=? ORDER BY seq DESC LIMIT ?",
                (experiment, limit),
            ).fetchall()
        return [
            dict(seq=r[0], at=r[1], kind=r[2], data=json.loads(r[3])) for r in rows
        ][::-1]
