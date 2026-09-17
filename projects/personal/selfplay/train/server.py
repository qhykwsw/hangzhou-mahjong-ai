"""Batched GPU inference over the shared request block."""

import time

import numpy as np
import torch

from selfplay_core.actions import N_ACTIONS


class InferenceServer:
    def __init__(self, req, device, twohot, cfg):
        self.req = req
        self.device = device
        self.twohot = twohot
        self.cfg = cfg
        rows = int(np.prod(req.lead))
        pin = device.type == "cuda"
        self.staging = {}
        for name, dtype in (("planes", torch.float32), ("scalars", torch.float32), ("events", torch.int64),
                            ("mask", torch.bool), ("oracle_planes", torch.uint8), ("oracle_scalars", torch.float32)):
            shape = (rows, *req[name].shape[len(req.lead):])
            t = torch.empty(shape, dtype=dtype, pin_memory=pin)
            self.staging[name] = (t, t.numpy())
        self.forward_calls = 0
        self.forward_rows = 0
        self.t_gather = self.t_forward = self.t_write = 0.0

    @torch.inference_mode()
    def serve(self, ready, current_model, opponent_models):
        """ready: list of (worker, n_requests).  Fills out_* fields in place."""
        req = self.req
        ws = np.concatenate([np.full(n, w, dtype=np.int64) for w, n in ready])
        is_ = np.concatenate([np.arange(n, dtype=np.int64) for _, n in ready])
        slots = req["slot"][ws, is_]
        greedy = req["greedy"][ws, is_].astype(bool)
        for slot in np.unique(slots):
            sel = slots == slot
            w_sel, i_sel = ws[sel], is_[sel]
            t0 = time.perf_counter()
            n = len(w_sel)

            def upload(name):
                t, arr = self.staging[name]
                np.take(req[name].reshape(-1, *req[name].shape[2:]), w_sel * req.lead[1] + i_sel, axis=0, out=arr[:n]) if arr.dtype == req[name].dtype else arr[:n].__setitem__(slice(None), req[name][w_sel, i_sel])
                return t[:n].to(self.device, non_blocking=True)

            planes = upload("planes")
            scalars = upload("scalars")
            events = upload("events")
            mask = upload("mask")
            # Inference runs in fp32: at these batch sizes the cost is kernel
            # launches, and autocast would re-cast every weight on every call.
            if slot == 0:
                op = upload("oracle_planes").float() / 255.0
                os_ = upload("oracle_scalars")
                logits, value_logits, _ = current_model(planes, scalars, events, op, os_)
                value = self.twohot.expectation(value_logits)
                g = torch.from_numpy(greedy[sel]).to(self.device)
            else:
                model = opponent_models[int(slot)]
                logits = model.actor_logits(planes, scalars, events)
                value = torch.zeros(n, device=self.device)
                g = torch.full((n,), self.cfg.opponent_sampling == "greedy", device=self.device)
            logits = logits.masked_fill(~mask, -1e9)
            logp = torch.log_softmax(logits, -1)
            # Gumbel-max sampling: two kernels instead of multinomial's many.
            gumbel = -torch.log(-torch.log(torch.rand_like(logp).clamp_(1e-10, 1.0)))
            sampled = (logp + gumbel).argmax(-1)
            best = logits.argmax(-1)
            action = torch.where(g, best, sampled)
            chosen_logp = logp.gather(1, action[:, None]).squeeze(1)
            out = torch.stack([action.float(), chosen_logp, value], 1).cpu().numpy()
            t1 = time.perf_counter()
            req["out_action"][w_sel, i_sel] = out[:, 0].astype(np.int16)
            req["out_logprob"][w_sel, i_sel] = out[:, 1]
            req["out_value"][w_sel, i_sel] = out[:, 2]
            t2 = time.perf_counter()
            self.t_forward += t1 - t0
            self.t_write += t2 - t1
            self.forward_calls += 1
            self.forward_rows += len(w_sel)

    def reset_stats(self):
        self.forward_calls = self.forward_rows = 0
        self.t_gather = self.t_forward = self.t_write = 0.0
