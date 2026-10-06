"""Sampling-hook decisions through CUDA graphs against the eager PG path (one to two minutes of GPU).

    python -m guided.probe_hook_graphs --actor <exported Actor dir> [--rows 3000] [--boost 6] [--seeds 20]

PG collection with a recipe's sample hook (e.g. boosted deviations) used to
run the learner eagerly in fp32 (collection.collect), and now replays a graph
of the rollout's fp16 snapshot's logits and runs the hook on them
(ActorGraphs.decide). On rows from random play with the Actor's own encoder,
this reports, by batch size (the largest beyond the graphs, run eagerly):

  logits      the largest absolute logit difference, graph against eager fp32
              and against eager fp16 (the snapshot without a graph)
  argmax      the share of rows whose argmax over the legal actions agrees
  actions     under one seed, the share of rows whose hook action agrees
  rates       over --seeds seeds, each path's deviation rate (action is not the
              argmax) and mean hook log weight, which must agree within noise
  ms          host milliseconds a batch: eager fp32 (batch, forward and hook)
              against the graph launch with its hook, read back

With --captured it also compares a capturable hook (the recipe's cumsum
version, marked ``capturable``, captured with the forward) against the same
hook run outside the graph, on the same rows and fp16 snapshot ("captured"):

  argmax      the share of rows whose argmax agrees (the same logits: 1.0)
  rates       over --seeds seeds, each path's deviation rate and mean log
              weight, which must agree within noise
  draws       a captured hook returning its uniform draws: two replays of one
              batch in a row draw differently, and reseeding repeats a draw
  ms          host milliseconds a batch, outside against inside the graph
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from . import bootstrap  # noqa: F401
from .deployment import load_actor
from .encoding import encoder
from .inference import ActorGraphs, hooked_decision, policy_batch, policy_logits, prepare_frozen


def boosted_deviations(boost):
    """A copy of the RL recipe's hook (rl/recipe/loop.py, 188 and 196): deviate boost times as often as the policy."""
    def sample(logits, mask):
        logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        probs = logp.exp()
        best = probs.argmax(-1)
        leave = (1.0 - probs.gather(1, best[:, None]).squeeze(1)).clamp_min(0.0)
        rate = (boost * leave).clamp(max=1.0)
        others = probs.scatter(1, best[:, None], 0.0)
        others = others / others.sum(-1, keepdim=True).clamp_min(1e-12)
        onehot = torch.zeros_like(probs).scatter(1, best[:, None], 1.0)
        mixed = (1.0 - rate)[:, None] * onehot + rate[:, None] * others
        actions = torch.multinomial(mixed, 1).squeeze(1)
        weight = (boost * leave / rate.clamp_min(1e-12)).clamp_min(1.0)
        return actions, weight.log()

    return sample


def cumsum_deviations(boost, capturable):
    """The RL line's graph-safe version of boosted_deviations (one uniform pair and a cumsum per row)."""
    def sample(logits, mask):
        probs = torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
        top, best = probs.max(-1)
        leave = (1.0 - top).clamp_min(0.0)
        rate = (boost * leave).clamp(max=1.0)
        u = 1.0 - torch.rand(len(probs), 2, device=probs.device)  # (0, 1]
        cdf = probs.scatter(1, best[:, None], 0.0).cumsum(-1)
        alt = (cdf < u[:, 1:] * cdf[:, -1:]).sum(-1).clamp(max=probs.shape[1] - 1)
        actions = torch.where(u[:, 0] <= rate, alt, best)
        weight = (boost * leave / rate.clamp_min(1e-12)).clamp_min(1.0)
        return actions, weight.log()

    if capturable:
        sample.capturable = True
    return sample


def uniform_draws(logits, mask):
    """A capturable hook that plays the argmax and returns its row's uniform draw in the logprob column."""
    return logits.masked_fill(~mask, -1e9).argmax(-1), torch.rand(len(logits), device=logits.device)


uniform_draws.capturable = True


def captured(graphs_out, graphs_in, frozen, rows, sizes, boost, seeds):
    """The capturable hook inside the graph against the same hook outside it, by batch size."""
    outside, inside = cumsum_deviations(boost, False), cumsum_deviations(boost, True)
    report = {}
    for size in sizes:
        batches = [rows[i:i + size] for i in range(0, len(rows) - size + 1, size)][:60]
        argmax = []
        for part in batches:
            out = graphs_out.decide("model:current", frozen, part, sample=True, hook=outside).result().copy()
            got = graphs_in.decide("model:current", frozen, part, sample=True, hook=inside).result().copy()
            argmax.append(float((out[:, 2] == got[:, 2]).mean()))
        rates = {"outside": [], "inside": []}
        weights = {"outside": [], "inside": []}
        for seed in range(seeds):
            for name, graphs, hook in (("outside", graphs_out, outside), ("inside", graphs_in, inside)):
                torch.manual_seed(2000 + seed)
                dev, logw = [], []
                for part in batches:
                    out = graphs.decide("model:current", frozen, part, sample=True, hook=hook).result().copy()
                    dev.append(out[:, 0] != out[:, 2])
                    logw.append(out[:, 1])
                rates[name].append(float(np.concatenate(dev).mean()))
                weights[name].append(float(np.concatenate(logw).mean()))
        timing = {}
        for name, graphs, hook in (("outside", graphs_out, outside), ("inside", graphs_in, inside)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for part in batches:
                graphs.decide("model:current", frozen, part, sample=True, hook=hook).result()
            torch.cuda.synchronize()
            timing[name] = round(1000 * (time.perf_counter() - start) / len(batches), 3)
        # Draws: the same batch replayed twice in a row, then again under one seed twice.
        part = batches[0]
        first = graphs_in.decide("draws", frozen, part, sample=True, hook=uniform_draws).result()[:, 1].copy()
        second = graphs_in.decide("draws", frozen, part, sample=True, hook=uniform_draws).result()[:, 1].copy()
        torch.manual_seed(77)
        seeded = graphs_in.decide("draws", frozen, part, sample=True, hook=uniform_draws).result()[:, 1].copy()
        torch.manual_seed(77)
        again = graphs_in.decide("draws", frozen, part, sample=True, hook=uniform_draws).result()[:, 1].copy()
        stat = lambda v: [round(float(np.mean(v)), 5), round(float(np.std(v) / np.sqrt(len(v))), 5)]  # noqa: E731
        report[size] = {
            "batches": len(batches),
            "argmax_agree": round(float(np.mean(argmax)), 5),
            "deviation_rate": {k: stat(v) for k, v in rates.items()},
            "mean_log_weight": {k: stat(v) for k, v in weights.items()},
            "draws_differ_between_replays": bool((first != second).any()),
            "draws_repeat_under_one_seed": bool(np.array_equal(seeded, again)),
            "draws_in_unit_interval": bool(((first >= 0) & (first < 1)).all()),
            "ms_per_batch": timing,
        }
    report["counts"] = {"outside": graphs_out.counts, "inside": graphs_in.counts}
    return report


def played_rows(count, encode, lookahead, seed):
    """Rows of real decisions (more than one legal action) from hands of uniformly random play."""
    from mahjong_ai.domain.engine import HandState, derive_seed
    from mahjong_ai.domain.types import RoundConfig

    rng = np.random.default_rng(seed)
    config, rows, hand = RoundConfig(rounds=16), [], 0
    while len(rows) < count:
        hand += 1
        state = HandState(config, derive_seed(seed, hand), hand % 4, 1 + hand % 16)
        while not state.settled and len(rows) < count:
            legal = state.legal_actions()
            if len(legal) > 1:
                rows.append(encode(state.observation(legal), lookahead))
            state.step(legal[int(rng.integers(len(legal)))], legal)
    return rows


def eager(model, rows, hook, device):
    batch = policy_batch(rows, device)
    logits = policy_logits(model, batch["planes"], batch["scalars"], batch["events"]).float()
    return logits, batch["mask"], hooked_decision(logits, batch["mask"], hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--boost", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--captured", action="store_true", help="also compare a capturable hook inside and outside the graph")
    parser.add_argument("--only-captured", action="store_true", help="only that comparison")
    args = parser.parse_args()
    device = torch.device("cuda")
    learner, meta = load_actor(args.actor, "cuda")
    frozen = prepare_frozen(copy.deepcopy(learner).requires_grad_(False), device)
    rows = played_rows(args.rows, encoder(meta["encoding"]), meta["config"]["lookahead"], 17)
    hook = boosted_deviations(args.boost)
    graphs = ActorGraphs()
    report = {"device": torch.cuda.get_device_name(0), "rows": len(rows), "boost": args.boost, "sizes": {}}
    with torch.no_grad():
        if args.captured or args.only_captured:
            report["captured"] = captured(ActorGraphs(), ActorGraphs(), frozen, rows, (1, 7, 46, 128), args.boost, args.seeds)
        for size in () if args.only_captured else (1, 7, 46, 128, 150):
            batches = [rows[i:i + size] for i in range(0, len(rows) - size + 1, size)][:60]
            stats = dict(logit_fp32=0.0, logit_fp16=0.0, argmax=[], actions_fp32=[], actions_fp16=[])
            for b, part in enumerate(batches):
                torch.manual_seed(b)
                logits32, mask, out32 = eager(learner, part, hook, device)
                torch.manual_seed(b)
                logits16, _, out16 = eager(frozen, part, hook, device)
                torch.manual_seed(b)
                got = graphs.decide("model:current", frozen, part, sample=True, hook=hook).result()
                # The graph's logits, replayed for the comparison (the last replay of this shape).
                shape = graphs.shape(part)
                entry = graphs.graphs.get(("model:current", *shape, (len(part[0]["planes"]), len(part[0]["scalars"])), "logits")) if shape else None
                graph_logits = entry[1][: len(part)] if entry else ActorGraphs._logits(frozen, *(policy_batch(part, device)[k] for k in ("planes", "scalars", "events", "mask")))
                legal = mask
                stats["logit_fp32"] = max(stats["logit_fp32"], float((graph_logits - logits32).abs().masked_fill(~legal, 0).max()))
                stats["logit_fp16"] = max(stats["logit_fp16"], float((graph_logits - logits16).abs().masked_fill(~legal, 0).max()))
                stats["argmax"].append(float((out32[:, 2].cpu().numpy() == got[:, 2]).mean()))
                stats["actions_fp32"].append(float((out32[:, 0].cpu().numpy() == got[:, 0]).mean()))
                stats["actions_fp16"].append(float((out16[:, 0].cpu().numpy() == got[:, 0]).mean()))
            rates = {"eager_fp32": [], "graph": []}
            weights = {"eager_fp32": [], "graph": []}
            for seed in range(args.seeds):
                for name in rates:
                    torch.manual_seed(1000 + seed)
                    dev, logw = [], []
                    for part in batches:
                        out = (eager(learner, part, hook, device)[2].cpu().numpy() if name == "eager_fp32"
                               else graphs.decide("model:current", frozen, part, sample=True, hook=hook).result().copy())
                        dev.append(out[:, 0] != out[:, 2])
                        logw.append(out[:, 1])
                    rates[name].append(float(np.concatenate(dev).mean()))
                    weights[name].append(float(np.concatenate(logw).mean()))
            timing = {}
            for name in ("eager_fp32", "graph"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for part in batches:
                    if name == "eager_fp32":
                        eager(learner, part, hook, device)[2].cpu()
                    else:
                        graphs.decide("model:current", frozen, part, sample=True, hook=hook).result()
                torch.cuda.synchronize()
                timing[name] = round(1000 * (time.perf_counter() - start) / len(batches), 3)
            report["sizes"][size] = {
                "batches": len(batches),
                "max_logit_diff_vs_eager_fp32": round(stats["logit_fp32"], 5),
                "max_logit_diff_vs_eager_fp16": round(stats["logit_fp16"], 5),
                "argmax_agree_vs_fp32": round(float(np.mean(stats["argmax"])), 5),
                "action_agree_same_seed_vs_fp32": round(float(np.mean(stats["actions_fp32"])), 5),
                "action_agree_same_seed_vs_fp16": round(float(np.mean(stats["actions_fp16"])), 5),
                "deviation_rate": {k: [round(float(np.mean(v)), 5), round(float(np.std(v) / np.sqrt(len(v))), 5)] for k, v in rates.items()},
                "mean_log_weight": {k: [round(float(np.mean(v)), 5), round(float(np.std(v) / np.sqrt(len(v))), 5)] for k, v in weights.items()},
                "ms_per_batch": timing,
            }
    report["graph_seconds"] = {k: round(v, 3) for k, v in graphs.seconds.items()}
    report["graph_counts"] = graphs.counts
    print(json.dumps(report))


if __name__ == "__main__":
    main()
