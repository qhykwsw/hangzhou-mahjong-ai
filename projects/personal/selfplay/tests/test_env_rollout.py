"""Table + SampleWriter + GAE invariants with a random legal-action stub model."""

import random

import numpy as np

from config import Config
from env import SLOT_CURRENT, Table
from opponents import BUILTIN, OpponentFactory
from rollout import FIELDS, SampleWriter, SharedBlock


def run_tables(n_tables=4, target=1500, seed=3, eval_mode=False, shaping=0.5):
    cfg = Config(tables_per_worker=n_tables, rounds_min=2, rounds_max=6, rounds_choices_prob=0.0)
    block = SharedBlock(FIELDS, (1, target + 64 * n_tables), create=True)
    writer = SampleWriter(block, 0, target + 64 * n_tables)
    factory = OpponentFactory(BUILTIN)
    rng = random.Random(seed)
    tables = [Table(cfg, random.Random(rng.getrandbits(62)), factory, writer, i, eval_mode) for i in range(n_tables)]
    league = {"fixed": factory.ids(), "fixed_weights": None, "slots": [], "slot_weights": [], "fixed_prob": 1.0}
    for t in tables:
        t.configure(league, shaping)
    rounds = 0
    while writer.count < target and rounds < 20000:
        for t in tables:
            r = t.advance()
            legal = r["legal"]
            action = int(rng.choice(list(legal)))
            t.apply(action, logprob=-1.0, value=float(rng.uniform(-5, 5)))
        rounds += 1
        if eval_mode and rounds > 400:
            break
    records = [rec for t in tables for rec in t.drain_records()]
    chains = [c for t in tables for c in t.chains_for_gae()]
    writer.compute_gae(chains, cfg.gamma, cfg.gae_lambda)
    return cfg, block, writer, records, tables, chains


def test_rollout_invariants():
    cfg, b, writer, records, tables, chains = run_tables()
    n = writer.count
    assert n >= 1500
    assert records, "at least one game must have finished"
    for rec in records:
        assert sum(rec["scores"]) == 0
        assert any(k == "model" for k, _ in rec["seats"])
    done = b["done"][0, :n]
    assert done.sum() >= 1
    # every chain ends either terminal or bootstrapped; advantages are finite
    assert np.isfinite(b["adv"][0, :n]).all() and np.isfinite(b["ret"][0, :n]).all()
    # actions are legal under their stored masks
    acts = b["action"][0, :n].astype(np.int64)
    assert b["mask"][0, np.arange(n), acts].all()
    # aux targets: every terminal transition knows its game outcome
    term = np.nonzero(done)[0]
    assert b["aux_game_known"][0, term].all()
    # rewards without shaping would be integer score deltas; with shaping they
    # are still finite and bounded by the biggest settlement plus shaping range
    assert np.abs(b["reward"][0, :n]).max() < 20000
    # events / planes stored quantized
    assert b["planes"].dtype == np.uint8 and b["events"].dtype == np.int8
    b.close(unlink=True)


def test_game_reward_telescopes_to_table_score_without_shaping():
    cfg, b, writer, records, tables, chains = run_tables(n_tables=2, target=800, seed=9, shaping=0.0)
    n = writer.count
    reward, done = b["reward"][0, :n], b["done"][0, :n]
    aux_game, known = b["aux_game"][0, :n], b["aux_game_known"][0, :n]
    exact, finished = 0, 0
    for indices, boot in chains:
        idx = np.asarray(indices)
        if done[idx[-1]]:
            finished += 1
            assert known[idx[-1]]
            # A finished chain covers one seat's decisions of one whole game, so
            # its rewards sum to that seat's final table score unless a hand
            # settled before the seat's very first decision (no transition yet).
            if abs(reward[idx].sum() - aux_game[idx[-1]]) < 1e-3:
                exact += 1
    assert finished >= 4
    assert exact >= 0.7 * finished, (exact, finished)
    # every stored sample belongs to some chain handed to GAE
    covered = sum(len(c) for c, _ in chains)
    assert covered == n
    b.close(unlink=True)


def test_eval_tables_produce_records_without_samples():
    cfg, b, writer, records, tables, chains = run_tables(n_tables=2, target=64, seed=5, eval_mode=True)
    assert writer.count == 0
    assert records and all(sum(k == "model" for k, _ in r["seats"]) == 1 for r in records)
    b.close(unlink=True)
