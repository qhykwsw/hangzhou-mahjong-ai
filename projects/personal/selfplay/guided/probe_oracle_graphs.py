"""Check the CUDA-graph path of the learner's oracle inputs (guided.oracle, ADR-0018) on a CUDA host.

    python -m guided.probe_oracle_graphs [--actor <oracle start dir>] [--rounds 48]

The CPU tests replace the graphs with a stand-in; this runs the real ones, as
collection.collect does (the fp16 precast snapshot, pinned buffers, two
launches in flight). Over batch sizes across the buckets, with and without a
sample hook, it compares ActorGraphs.decide(..., oracle=True) with the eager
fp16 decision fed the same oracle (exact batch, no padding):

  argmax_agree     share of rows whose argmax is the eager one (1.0 up to fp16 ties)
  logprob_max_err  the chosen action's logprob against the eager one (padding rounding: ~1e-3)
  refreshed        share of rows of a second launch on the same graph, with other
                   oracle values, that follow them (the graph reads its static
                   oracle buffers; nothing was baked in at capture)
  plain_agree      share of rows a decision without oracle on the same rows
                   (a teacher's) decides as the plain eager one
  oracle_used      the oracle changes some decisions
  pinned           the oracle's host buffers are page-locked
  oracle_graphs    graphs captured with the oracle inputs

and prints one JSON line; "ok" is their conjunction.
"""

import argparse
import copy
import json

import numpy as np
import torch

from . import bootstrap  # noqa: F401
from .inference import ActorGraphs, policy_batch, prepare_frozen
from .oracle import PLANES, SCALARS
from .probe_graphs import rows_for


def with_oracle(rows, rng, share=0.8):
    """``rows`` with random masked oracle fields on about ``share`` of them."""
    out = []
    for row in rows:
        row = {k: v for k, v in row.items() if k not in (PLANES, SCALARS)}
        if rng.random() < share:
            row[PLANES] = (rng.random((16, 34)) < 0.15).astype(np.float16)
            row[SCALARS] = (rng.random(4) * (rng.random(4) < 0.5)).astype(np.float16)
        out.append(row)
    return out


def eager(model, rows, device, oracle):
    batch = policy_batch(rows, device, oracle=oracle)
    args = (batch["planes"], batch["scalars"], batch["events"], batch["mask"])
    return ActorGraphs._decide(model, *args, False, **({"oracle": batch["oracle"]} if oracle else {})).cpu().numpy()


def check(model, graphs, device, rounds, seed=11):
    """The report (without "device" and "pinned") of ``rounds`` launches on ``graphs``."""
    from selfplay_core.actions import N_ACTIONS

    planes, scalars = model.encoder.tile_in.in_features, model.encoder.state_in.in_features
    rng = np.random.default_rng(seed)

    def argmax_hook(logits, mask):
        return logits.masked_fill(~mask, -1e9).argmax(-1), None

    counts = dict.fromkeys(("rows", "argmax", "refreshed", "plain", "changed"), 0)
    error = 0.0
    with torch.no_grad():
        for r in range(rounds):
            size = (1, 7, 46, 128)[r % 4]
            rows = with_oracle(rows_for(rng, size, planes, scalars, N_ACTIONS), rng)
            other = with_oracle(rows, rng)
            hook = {"hook": argmax_hook} if r % 2 else {}
            # Two launches on one graph before either is read, as two rounds in flight.
            launched = [graphs.decide("current", model, batch, sample=bool(hook), oracle=True, **hook)
                        for batch in (rows, other)]
            plain = graphs.decide("teacher", model, rows, sample=False)
            got, again = (decided.result().copy() for decided in launched)
            blind = plain.result().copy()
            want, want_other = eager(model, rows, device, True), eager(model, other, device, True)
            counts["rows"] += size
            counts["argmax"] += int((got[:, 2] == want[:, 2]).sum())
            counts["refreshed"] += int((again[:, 2] == want_other[:, 2]).sum())
            counts["plain"] += int((blind[:, 2] == eager(model, rows, device, False)[:, 2]).sum())
            counts["changed"] += int((blind[:, 2] != got[:, 2]).sum())
            error = max(error, float(np.abs(got[:, 1] - want[:, 1]).max()))
    n = counts["rows"]
    return {"rows": n, "argmax_agree": counts["argmax"] / n, "logprob_max_err": error,
            "refreshed": counts["refreshed"] / n, "plain_agree": counts["plain"] / n,
            "oracle_used": counts["changed"] > 0,
            "oracle_graphs": sum(len(key[3]) == 3 for key in graphs.graphs)}


def passed(report):
    return (report["argmax_agree"] >= 0.99 and report["refreshed"] >= 0.99 and report["plain_agree"] >= 0.99
            and report["logprob_max_err"] < 1e-2 and report["oracle_used"] and report["oracle_graphs"] > 0
            and report.get("pinned", True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", help="an Actor with oracle inputs (expand --oracle-inputs); default a fresh one")
    parser.add_argument("--rounds", type=int, default=48)
    args = parser.parse_args()
    from .networks import Actor, with_oracle_inputs

    if args.actor:
        from .deployment import load_actor

        actor, _ = load_actor(args.actor, "cuda", oracle=True)
    else:
        actor = with_oracle_inputs(Actor(128, 4, 4)).cuda()
    with torch.no_grad():  # trained-looking projections, so the oracle matters
        torch.nn.init.normal_(actor.encoder.oracle_tile_in.weight, std=0.5)
        torch.nn.init.normal_(actor.encoder.oracle_state_in.weight, std=0.5)
    # The collection's rollout snapshot (collection.collect, precast_collection).
    model = prepare_frozen(copy.deepcopy(actor).eval().requires_grad_(False), "cuda")
    graphs = ActorGraphs()
    report = {"device": torch.cuda.get_device_name(0)} | check(model, graphs, "cuda", args.rounds)
    rings = [ring for name, ring in graphs.hosts.items() if name[0] != "answers" and len(name[3]) == 3]
    report["pinned"] = bool(rings) and all(t.is_pinned() for ring in rings for entry in ring for t in entry[0])
    report["ok"] = passed(report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
