"""Training the outcome head alone (cfg["outcome_only"]; survey item 4, the stage re-rank).

The start is 227 with a zero-initialized outcome head (guided.expand
--outcome-head): per action, the distribution of the learner's score in the
rest of the hand over the hand's possible settlements (guided.outcome bins).
Only the maintainer's stage re-rank reads it, at inference, among the
policy's own near-ties. Here the policy never moves: the rollouts are 227's
own vine games and only the head trains (return_outcome="detached", an
optimizer over its parameters alone), so the exported policy is 227 bit for
bit and a stage comparison with and without the re-rank measures the re-rank
alone.

Each row gives the head the action the hand played and the hand's
settlement (the sum of its rows' rewards; a hand pays only when it ends); a
vine deviation row gives two: its argmax (what the hand played, recomputed
with the unchanged policy) with the settlement, and the deviation with its
copy's playout. Scores are raw, no luck subtracted: the re-rank needs the
outcome's distribution, not a lower-variance mean. Before a rollout trains
the head, its held-out cross entropy is measured against the bins' marginal.
"""

import numpy as np
import torch

from .batch import actor_args, make_batch, masked_log_probs


class OutcomeTrainer:
    def __init__(self, actor, spec, device):
        from guided.networks import outcome_parameters

        if getattr(actor, "outcome_spec", None) is None:
            raise ValueError("outcome_only needs a start with an outcome head (guided.expand --outcome-head)")
        self.device = device
        self.bins = list(actor.outcome_spec["bins"])
        self.size = int(spec.get("microbatch", 512))
        self.epochs = int(spec.get("epochs", 1))
        self.probe = int(spec.get("probe", 20000))
        self.opt = torch.optim.Adam(outcome_parameters(actor), lr=float(spec.get("lr", 3e-4)), eps=1e-5)
        self.marginal = np.ones(len(self.bins))  # running counts of the bins, the baseline's distribution

    def pairs(self, actor, rows):
        """(row index, action, score) for every row and every deviation."""
        hand = np.array([r["hand"] for r in rows])
        if "game_mark" in rows[0]:
            mark = np.array([r["game_mark"] for r in rows])
            game = np.cumsum(np.r_[1, (np.diff(mark) != 0).astype(int)]) - 1
        else:
            game = np.cumsum(np.r_[1, (np.diff(hand) < 0).astype(int)]) - 1
        new_hand = np.r_[True, (np.diff(hand) != 0) | (np.diff(game) != 0)]
        hand_id = np.cumsum(new_hand) - 1
        settle = np.bincount(hand_id, weights=np.array([float(r["reward"]) for r in rows]))[hand_id]
        deviated = np.flatnonzero([bool(r.get("deviated")) and "playout" in r for r in rows])
        actions = np.array([int(r["action"]) for r in rows])
        main = actions.copy()
        if len(deviated):
            # The hand played the argmax at a deviation row; the policy is the collecting one.
            actor.eval()
            with torch.no_grad():
                for s in range(0, len(deviated), 1024):
                    b = deviated[s:s + 1024]
                    mb = make_batch([rows[i] for i in b], self.device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(self.device).type == "cuda"):
                        logits = actor(*actor_args(mb))
                    main[b] = masked_log_probs(logits, mb["mask"]).argmax(-1).cpu().numpy()
        index = np.r_[np.arange(len(rows)), deviated]
        action = np.r_[main, actions[deviated]]
        score = np.r_[settle, [float(rows[i]["playout"]) for i in deviated]]
        return index, action, score, len(deviated)

    def _logits(self, actor, rows, index, action):
        mb = make_batch([rows[i] for i in index], self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(self.device).type == "cuda"):
            _, outcome = actor(*actor_args(mb), return_outcome="detached")
        a = torch.as_tensor(action, device=self.device)
        return outcome.float().gather(1, a[:, None, None].expand(-1, 1, outcome.shape[-1]))[:, 0]

    def step(self, actor, rows, rng):
        from guided.outcome import bin_index, outcome_loss

        index, action, score, deviations = self.pairs(actor, rows)
        target = bin_index(score, self.bins)
        stats = {"outcome_pairs": len(index), "outcome_deviations": deviations,
                 "outcome_zero_share": float((target == bin_index(0.0, self.bins)).mean())}
        actor.train(False)
        # Held out: this rollout's pairs before the head sees them, against the bins' marginal so far.
        probe = rng.choice(len(index), size=min(len(index), self.probe), replace=False)
        nll, expect = [], []
        values = torch.as_tensor(self.bins, device=self.device)
        with torch.no_grad():
            for s in range(0, len(probe), 1024):
                p = probe[s:s + 1024]
                logp = torch.log_softmax(self._logits(actor, rows, index[p], action[p]), -1)
                t = torch.as_tensor(target[p], device=self.device)
                nll.append(-logp.gather(1, t[:, None])[:, 0])
                expect.append((logp.exp() * values).sum(-1))
        nll = torch.cat(nll).cpu().numpy()
        expect = torch.cat(expect).cpu().numpy()
        base = self.marginal / self.marginal.sum()
        stats |= {"outcome_nll": float(nll.mean()), "outcome_nll_marginal": float(-np.log(base[target[probe]]).mean()),
                  "outcome_mean_corr": float(np.corrcoef(expect, score[probe])[0, 1]) if expect.std() > 0 else 0.0}
        self.marginal += np.bincount(target, minlength=len(self.bins))
        losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(index))
            for s in range(0, len(order), self.size):
                b = order[s:s + self.size]
                mb = make_batch([rows[i] for i in index[b]], self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(self.device).type == "cuda"):
                    _, outcome = actor(*actor_args(mb), return_outcome="detached")
                loss = outcome_loss(outcome.float(), torch.as_tensor(action[b], device=self.device),
                                    torch.as_tensor(score[b], device=self.device), self.bins)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                losses.append(loss.detach())
        actor.zero_grad(set_to_none=True)
        stats["outcome_loss_last"] = float(torch.stack(losses[-50:]).mean()) if losses else 0.0
        return stats
