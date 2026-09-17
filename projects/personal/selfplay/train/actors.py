"""Environment worker process: T tables, batched model requests over shared memory."""

import os
import random
import time
import traceback

import numpy as np

from config import Config
from env import Table
from opponents import OpponentFactory
from rollout import FIELDS, REQUEST_FIELDS, SampleWriter, SharedBlock


def fill_request(req, w, i, r):
    req["planes"][w, i] = r["planes"]
    req["scalars"][w, i] = r["scalars"]
    req["events"][w, i] = r["events"]
    req["mask"][w, i] = r["mask"]
    req["slot"][w, i] = r["slot"]
    req["greedy"][w, i] = 1 if r["greedy"] else 0
    req["table"][w, i] = r["table"]
    if "oracle_planes" in r:
        req["oracle_planes"][w, i] = r["oracle_planes"]
        req["oracle_scalars"][w, i] = r["oracle_scalars"]


def worker_main(worker_id, role, cfg_dict, layout, conn, entries, seed):
    try:
        _worker(worker_id, role, cfg_dict, layout, conn, entries, seed)
    except Exception:
        conn.send(("error", worker_id, traceback.format_exc()))
    finally:
        conn.close()


def _worker(worker_id, role, cfg_dict, layout, conn, entries, seed):
    cfg = Config(**cfg_dict)
    try:
        os.nice(cfg.worker_nice)
    except OSError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    T = cfg.tables_per_worker
    req = SharedBlock(REQUEST_FIELDS, layout["req_lead"], create=False, names=layout["req_names"])
    store = writer = None
    if role == "train":
        store = SharedBlock(FIELDS, layout["store_lead"], create=False, names=layout["store_names"])
        writer = SampleWriter(store, worker_id, layout["capacity"])
    factory = OpponentFactory(entries)
    rng = random.Random(seed)
    tables = [
        Table(cfg, random.Random(rng.getrandbits(62)), factory, writer, i, eval_mode=(role == "eval"))
        for i in range(T)
    ]
    w = worker_id

    def one_round(records):
        for i, t in enumerate(tables):
            if t.request is None:
                r = t.advance()
                fill_request(req, w, i, r)
        for t in tables:
            records.extend(t.drain_records())
        conn.send(("req", T, records))
        records.clear()
        msg = conn.recv()
        if msg[0] != "act":
            return msg
        for i, t in enumerate(tables):
            t.apply(int(req["out_action"][w, i]), float(req["out_logprob"][w, i]), float(req["out_value"][w, i]))
        return None

    records = []
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "collect":
            _, target, league, shaping = msg
            for t in tables:
                t.configure(league, shaping)
            if role == "train":
                writer.reset()
                started = time.perf_counter()
                while writer.count < target:
                    other = one_round(records)
                    if other is not None:
                        if other[0] == "stop":
                            return
                        raise RuntimeError(f"unexpected message during collection: {other[0]}")
                chains = [c for t in tables for c in t.chains_for_gae()]
                writer.compute_gae(chains, cfg.gamma, cfg.gae_lambda)
                conn.send(("done", writer.count, time.perf_counter() - started, records))
                records = []
            else:
                # eval workers serve requests until told to collect again or stop;
                # the parent ends an eval segment by sending ("pause",).
                while True:
                    other = one_round(records)
                    if other is None:
                        continue
                    if other[0] == "stop":
                        return
                    if other[0] == "pause":
                        conn.send(("paused",))
                        break
                    raise RuntimeError(f"unexpected message in eval loop: {other[0]}")
        elif kind == "pause":
            conn.send(("paused",))
