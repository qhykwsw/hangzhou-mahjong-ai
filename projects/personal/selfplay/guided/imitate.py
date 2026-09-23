"""Pure imitation of a frozen rule teacher, with phases switched by the data.

1. bc: the learner seat plays the teacher's action (behaviour cloning on
   teacher trajectories) at the peak learning rate.
2. dagger: the learner plays its own sampled action and the teacher labels
   every decision, still at the peak rate. Behaviour cloning alone never visits
   the states the student's own mistakes lead to.
3. decay: dagger continues while the rate falls linearly to the floor over
   decay_fraction of the time spent so far; then the run stops.

A phase ends when the teacher cross entropy on each rollout's fresh rows,
measured before those rows are trained on, has stopped improving (Plateau).
Snapshots are saved every snapshot_seconds for evaluation after training.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import time

from . import bootstrap
from .config import configuration
from .run import atomic_json

SCHEDULE = {
    "peak": 1e-4,
    "floor": 1e-6,
    "warmup_updates": 3,
    "window_seconds": 1800,
    "tolerance": 0.005,
    "min_phase_seconds": 1800,
    "decay_fraction": 0.35,
    "snapshot_seconds": 600,
}


class Plateau:
    """Relative improvement of the latest third of a window over its first third."""

    def __init__(self, window, tolerance, min_seconds):
        self.window, self.tolerance, self.min_seconds = window, tolerance, min_seconds
        self.points, self.began = [], None

    def reset(self, now):
        self.points, self.began = [], now

    def observe(self, now, value):
        self.points.append((now, value))

    def reached(self, now):
        if self.began is None or now - self.began < max(self.min_seconds, self.window):
            return False
        third = self.window / 3
        early = [v for t, v in self.points if now - self.window <= t < now - 2 * third]
        late = [v for t, v in self.points if t >= now - third]
        if not early or not late:
            return False
        before, after = sum(early) / len(early), sum(late) / len(late)
        return (before - after) / max(abs(before), 1e-9) < self.tolerance


def rate(schedule, phase, updates, now, decay_start, decay_end):
    peak = schedule["peak"]
    if updates < schedule["warmup_updates"]:
        return peak * (updates + 1) / (schedule["warmup_updates"] + 1)
    if phase != "decay":
        return peak
    progress = min(1.0, (now - decay_start) / max(1e-9, decay_end - decay_start))
    return peak + (schedule["floor"] - peak) * progress


def fresh_metrics(actor, rows, micro, device):
    """Teacher cross entropy and top-1 agreement on rows not yet trained on."""
    import torch
    from .learning import make_batch, masked_log_probs

    total_ce = agree = 0.0
    actor.eval()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["events"]))
    with torch.no_grad():
        for s in range(0, len(order), 4 * micro):
            mb = make_batch([rows[i] for i in order[s : s + 4 * micro]], device)
            logp = masked_log_probs(actor(mb["planes"], mb["scalars"], mb["events"]), mb["mask"])
            total_ce -= float(logp.gather(1, mb["teacher"][:, None]).sum())
            agree += float((logp.argmax(-1) == mb["teacher"]).sum())
    return total_ce / len(rows), agree / len(rows)


def train(run_dir, cfg, schedule, bundle_path, *, deadline, resume=False):
    import numpy as np
    import torch
    from .artifacts import read_bundle
    from .collection import collect
    from .learning import update
    from .networks import Actor, Critic

    code_id = bootstrap.verify_source()
    run_dir = Path(run_dir)
    latest = run_dir / "latest.pt"
    if latest.exists() != resume:
        raise FileExistsError("use --resume exactly when the run already exists")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU training")
    bundle = read_bundle(bundle_path)
    bundle_path = Path(bundle["manifest_path"])
    actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
    critic = Critic().to(device)  # Only feeds collection's value field; never trained.
    opt = torch.optim.Adam(actor.parameters(), lr=schedule["peak"], eps=1e-5)
    idle = torch.optim.Adam(critic.parameters())  # update() clears it; bc_only never steps it.
    replay = deque(maxlen=cfg["replay_capacity"])
    plateau = Plateau(schedule["window_seconds"], schedule["tolerance"], schedule["min_phase_seconds"])
    s = dict(phase="bc", updates=0, iteration=0, started=time.time(), decay_start=None,
             decay_end=None, next_snapshot=0.0, collected=0, optimized=0, phase_began=None)
    if resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["config"] != cfg or state["schedule"] != schedule or state["bundle"] != bundle["id"]:
            raise ValueError("resume must keep the recorded configuration, schedule and teacher")
        actor.load_state_dict(state["actor"])
        opt.load_state_dict(state["optimizer"])
        replay.extend(state["replay"])
        s = state["progress"]
        # Downtime does not count as training time.
        shift = time.time() - state["saved"]
        for key in ("started", "decay_start", "decay_end", "phase_began"):
            if s[key] is not None:
                s[key] += shift
        plateau.began = s["phase_began"]
        plateau.points = [(t + shift, v) for t, v in state["plateau"]]
        rng.bit_generator.state = state["numpy_rng"]
    else:
        plateau.reset(s["started"])
        s["phase_began"] = s["started"]
        atomic_json(run_dir / "config.json", {"config": cfg, "schedule": schedule,
                                              "bundle": bundle["id"], "teacher": bundle["teacher"],
                                              "code_id": code_id})

    def log(kind, **kwargs):
        row = {"kind": kind, "time": time.time(), **kwargs}
        with (run_dir / "log.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def save():
        tmp = latest.with_suffix(".tmp")
        torch.save({"format": "guided-imitation-v1", "code_id": code_id, "config": cfg,
                    "schedule": schedule, "bundle": bundle["id"], "actor": actor.state_dict(),
                    "optimizer": opt.state_dict(), "replay": list(replay), "progress": s,
                    "plateau": plateau.points, "saved": time.time(),
                    "numpy_rng": rng.bit_generator.state}, tmp)
        os.replace(tmp, latest)

    def snapshot(name):
        path = run_dir / "snapshots" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(elapsed_seconds=time.time() - s["started"], phase=s["phase"],
                    updates=s["updates"], collected_samples=s["collected"],
                    optimized_samples=s["optimized"])
        # guided-v1 with the Actor config: load_actor/export accept it directly.
        torch.save({"format": "guided-v1", "actor": actor.state_dict(), "config": cfg, **meta},
                   path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        atomic_json(path.with_suffix(".json"), meta)
        log("snapshot", path=str(path), **meta)

    last_save = time.time()
    if not resume:
        snapshot("seconds-000000")
        s["next_snapshot"] = schedule["snapshot_seconds"]
    log("started", resume=resume, phase=s["phase"], teacher=bundle["teacher"])
    try:
        while time.time() < deadline:
            now = time.time()
            if s["phase"] == "decay" and now >= s["decay_end"]:
                break
            s["iteration"] += 1
            teacher_plays = s["phase"] == "bc"
            began = time.time()
            rows, records, resources = collect(
                actor, critic, cfg | {"bc_only": teacher_plays}, bundle_path, bundle_path,
                {"kind": "collect", "iteration": s["iteration"], "deadline": deadline,
                 "labels": True, "history": []},
            )
            rows = [r for r in rows if r["teacher"] >= 0]
            if not rows:
                break
            collected = time.time() - began
            ce, agree = fresh_metrics(actor, rows, cfg["minibatch"], device)
            plateau.observe(time.time(), ce)
            lr = rate(schedule, s["phase"], s["updates"], time.time(), s["decay_start"], s["decay_end"])
            for group in opt.param_groups:
                group["lr"] = lr
            stats = update(actor, critic, opt, idle, rows,
                           cfg | {"bc_only": True, "deadline": deadline}, 1.0, list(replay), rng)
            replay.extend({k: r[k] for k in ("planes", "scalars", "events", "mask", "teacher",
                                             "label_version")} for r in rows)
            s["updates"] += 1
            s["collected"] += len(rows)
            s["optimized"] += stats["optimized_samples"]
            log("update", phase=s["phase"], lr=lr, fresh_ce=ce, fresh_agreement=agree,
                rows=len(rows), games=len(records), collect_seconds=collected,
                learn_seconds=time.time() - began - collected, sgd_steps=stats["sgd_steps"],
                teacher_loss=stats["teacher_loss"],
                elapsed=time.time() - s["started"], updates=s["updates"])
            now = time.time()
            if s["phase"] != "decay" and plateau.reached(now):
                done = s["phase"]
                if done == "bc":
                    s["phase"] = "dagger"
                else:
                    elapsed = now - s["started"]
                    s["phase"] = "decay"
                    s["decay_start"], s["decay_end"] = now, now + schedule["decay_fraction"] * elapsed
                snapshot(f"phase-end-{done}")
                plateau.reset(now)
                s["phase_began"] = now
                log("phase", ended=done, started=s["phase"], decay_end=s["decay_end"])
            while now - s["started"] >= s["next_snapshot"]:
                snapshot(f"seconds-{int(s['next_snapshot']):06d}")
                s["next_snapshot"] += schedule["snapshot_seconds"]
            if now - last_save >= cfg["checkpoint_seconds"]:
                save()
                last_save = now
        snapshot("final")
        save()
        atomic_json(run_dir / "status.json", {"state": "complete", "finished": time.time(),
                                              "phase": s["phase"], "updates": s["updates"]})
    except BaseException as exc:
        atomic_json(run_dir / "status.json", {"state": "failed", "error": str(exc)[-4000:],
                                              "time": time.time()})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--schedule", help="JSON overrides of SCHEDULE")
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = configuration(**json.loads(Path(args.config).read_text()))
    schedule = SCHEDULE | (json.loads(args.schedule) if args.schedule else {})
    if unknown := set(schedule) - SCHEDULE.keys():
        raise ValueError(f"unknown schedule keys: {sorted(unknown)}")
    train(args.run, cfg, schedule, args.bundle, deadline=args.deadline, resume=args.resume)


if __name__ == "__main__":
    main()
