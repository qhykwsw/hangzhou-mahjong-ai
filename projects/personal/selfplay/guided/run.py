"""New rule-guided experiment runner. No legacy checkpoint compatibility."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import shutil
import time

from . import bootstrap  # noqa: F401
from .config import configuration


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def pending_bundle(path, active):
    """A failed/incompatible arrival never replaces the active teacher."""
    from .artifacts import read_bundle

    if not path or not Path(path).exists():
        return active
    new, old = read_bundle(path), read_bundle(active)
    candidate = Path(new["manifest_path"])
    # Keep this experiment's frozen engine/protocol. New engine means a new experiment.
    if new["runtime_hash"] != old["runtime_hash"]:
        raise ValueError(
            "new teacher has a different engine runtime; retain the current teacher"
        )
    return candidate


def train(
    run_dir,
    cfg,
    bundle_path,
    panel_path,
    *,
    deadline,
    initial=None,
    resume=False,
    carry=False,
    updates=None,
    bc_seconds=0,
    eval_enabled=True,
    max_updates=None,
    snapshot_seconds=None,
    reference=None,
):
    import numpy as np
    import torch
    from .artifacts import read_bundle
    from .collection import collect
    from .evaluation import evaluate
    from .learning import update
    from .networks import Actor, Critic
    from .statistics import interval

    code_id = bootstrap.verify_source()
    if snapshot_seconds is not None and (snapshot_seconds <= 0 or resume or carry):
        raise ValueError("timed snapshots require a new run and a positive interval")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / "latest.pt"
    if existing.exists() and not resume:
        raise FileExistsError(f"run already exists: {run_dir}")
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU training")
    device = torch.device(cfg["device"])
    actor = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
    critic = Critic().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["critic_learning_rate"], eps=1e-5
    )
    replay = deque(maxlen=cfg["replay_capacity"])
    history_paths = []
    best_history = None
    generation = iteration = 0
    w, best = 1.0, None
    started = time.time()
    bc_until = started + bc_seconds
    next_eval = started + cfg["evaluation_seconds"]
    active = Path(read_bundle(bundle_path)["manifest_path"])
    panel_path = Path(read_bundle(panel_path)["manifest_path"])
    source = existing if resume else initial
    reference = str(Path(reference).resolve()) if reference else None
    if source:
        state = torch.load(source, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        if resume:
            recorded = state.get("reference")
            if reference is not None and reference != recorded:
                raise ValueError("resume must keep the recorded evaluation reference")
            reference = recorded
        if resume or carry:
            if state["config"] != cfg:
                raise ValueError("resume must use the recorded configuration")
            if read_bundle(state["panel"])["id"] != read_bundle(panel_path)["id"]:
                raise ValueError(
                    "resume/carry cannot change the frozen evaluation panel"
                )
            actor_opt.load_state_dict(state["actor_optimizer"])
            critic_opt.load_state_dict(state["critic_optimizer"])
            iteration, generation = state["iteration"], state["generation"]
            w, best = state["w"], state["best_score"]
            best_history = state.get("best_history")
            active = Path(state["bundle"])
            if resume:
                started, bc_until = state["started"], state["bc_until"]
                deadline = min(deadline, state["deadline"])
                next_eval = state["next_eval"]
            history_paths = state["history"]
            replay.extend(state["replay"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            if carry and best is not None:
                inherited_best = Path(source).parent / "best.pt"
                if not inherited_best.exists():
                    inherited_best = (
                        Path(best_history) if best_history else inherited_best
                    )
                if not inherited_best.is_file():
                    raise FileNotFoundError(
                        "carried run has a best score but no best checkpoint"
                    )
                shutil.copy2(inherited_best, run_dir / "best.pt")
    # A fixed Actor (e.g. the imitation starting point) evaluated on the same walls
    # as every generation, so improvement over it is measured as a paired difference.
    references = {}
    if reference:
        ref = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
        ref.load_state_dict(
            torch.load(reference, map_location=device, weights_only=False)["actor"]
        )
        references["model:reference"] = ref.eval()
    atomic_json(run_dir / "config.json", cfg)
    read_bundle(active)
    read_bundle(panel_path)

    def log(kind, **kwargs):
        row = {"kind": kind, "time": time.time(), "iteration": iteration, **kwargs}
        with (run_dir / "log.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    def save(path):
        state = {
            "format": "guided-v1",
            "code_id": code_id,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_opt.state_dict(),
            "critic_optimizer": critic_opt.state_dict(),
            "config": cfg,
            "iteration": iteration,
            "generation": generation,
            "w": w,
            "best_score": best,
            "bundle": str(active),
            "panel": str(Path(panel_path).resolve()),
            "started": started,
            "deadline": deadline,
            "bc_until": bc_until,
            "next_eval": next_eval,
            "history": history_paths,
            "replay": list(replay),
            "best_history": best_history,
            "reference": reference,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    progress = dict(updates=0, collected_samples=0, optimized_samples=0, sgd_steps=0)
    next_snapshot = snapshot_seconds

    def snapshot(name, scheduled_seconds):
        if snapshot_seconds is None:
            return
        path = run_dir / "snapshots" / (name + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"immutable snapshot already exists: {path}")
        metadata = dict(
            format="guided-actor-snapshot-v1",
            code_id=code_id,
            config=cfg,
            bundle=str(active),
            elapsed_seconds=time.time() - started,
            scheduled_seconds=scheduled_seconds,
            **progress,
        )
        tmp = path.with_suffix(".tmp")
        torch.save(metadata | {"actor": actor.state_dict()}, tmp)
        os.replace(tmp, path)
        atomic_json(path.with_suffix(".json"), metadata)
        log("timed_snapshot", path=str(path), **metadata)

    if not source:
        save(run_dir / "initial.pt")
    # Keep a healthy recovery point before any update can partially fail.
    save(existing)
    last_save = time.time()
    completed_updates = 0
    log("started", deadline=deadline, bundle=str(active), bc_seconds=bc_seconds)
    snapshot("initial", 0)
    try:
        while time.time() < deadline:
            iteration += 1
            bc = time.time() < bc_until
            if not bc and bc_until > started and not (run_dir / "bc-final.pt").exists():
                save(run_dir / "bc-final.pt")
                log("bc_checkpoint_saved", checkpoint=str(run_dir / "bc-final.pt"))
            phase_deadline = min(deadline, bc_until) if bc else deadline
            iteration_cfg = cfg | {"bc_only": bc, "deadline": phase_deadline}
            slots = list(
                rng.choice(history_paths, min(3, len(history_paths)), replace=False)
            )
            opponents = {}
            for path in slots:
                old = torch.load(path, map_location=device, weights_only=False)
                model = Actor(cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
                model.load_state_dict(old["actor"])
                model.eval()
                opponents["model:" + path] = model
            command = {
                "kind": "collect",
                "iteration": iteration,
                "deadline": phase_deadline,
                "labels": bc or w > 0,
                "history": list(opponents),
            }
            rows, records, resources = collect(
                actor, critic, iteration_cfg, active, panel_path, command, opponents
            )
            del opponents
            if bc and time.time() >= bc_until:
                log("bc_phase_complete", resources=resources)
                continue
            if not rows or time.time() >= deadline:
                log("collection_stopped", resources=resources)
                break
            learn_start = time.time()
            stats = update(
                actor,
                critic,
                actor_opt,
                critic_opt,
                rows,
                iteration_cfg,
                1.0 if bc else w,
                list(replay),
                rng,
            )
            for row in rows:
                if row["teacher"] >= 0:
                    replay.append(
                        {
                            k: row[k]
                            for k in (
                                "planes",
                                "scalars",
                                "events",
                                "mask",
                                "teacher",
                                "label_version",
                            )
                        }
                    )
            log(
                "update",
                phase="bc" if bc else "joint",
                w=w,
                teacher=read_bundle(active, verify=False)["teacher"],
                resources=resources,
                learning_seconds=time.time() - learn_start,
                stats=stats,
            )
            completed_updates += 1
            progress["updates"] += 1
            progress["collected_samples"] += len(rows)
            for key in ("optimized_samples", "sgd_steps"):
                progress[key] += stats.get(key, 0)
            if next_snapshot is not None:
                while time.time() - started >= next_snapshot:
                    snapshot(f"seconds-{next_snapshot:06d}", next_snapshot)
                    next_snapshot += snapshot_seconds
            if max_updates is not None and completed_updates >= max_updates:
                break
            if eval_enabled and time.time() >= next_eval and time.time() < deadline:
                candidate = active
                try:
                    candidate = pending_bundle(updates, active)
                except (ValueError, OSError, KeyError) as exc:
                    log("teacher_rejected", reason=str(exc))
                generation += 1

                def assess(bundle):
                    targets = ["model:current", read_bundle(bundle)["teacher"]]
                    return evaluate(
                        actor,
                        critic,
                        cfg,
                        bundle,
                        panel_path,
                        generation,
                        deadline,
                        targets=targets + list(references),
                        history=references,
                    )

                try:
                    result = assess(candidate)
                except (ValueError, TimeoutError, RuntimeError) as exc:
                    if candidate != active and time.time() < deadline:
                        log(
                            "teacher_rejected",
                            reason=str(exc),
                            candidate=str(candidate),
                        )
                        result = assess(active)
                        candidate = active
                    elif time.time() < deadline:
                        raise
                    else:
                        log("evaluation_incomplete_at_deadline")
                        break
                if references:
                    result["reference_minus_student"] = interval(
                        np.asarray(result["results"]["model:reference"]["group_scores"])
                        - np.asarray(result["results"]["model:current"]["group_scores"])
                    )
                atomic_json(run_dir / "evaluation" / f"{generation:04d}.json", result)
                active, w = candidate, result["w"]
                score = result["results"]["model:current"]["mean"]
                # Evaluation wall time determines the next interval, not the score.
                next_eval = time.time() + max(
                    cfg["evaluation_seconds"], 9 * result["resources"]["seconds"]
                )
                frozen = run_dir / "history" / f"{generation:04d}.pt"
                frozen.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"format": "guided-v1", "actor": actor.state_dict(), "config": cfg},
                    frozen,
                )
                history_paths.append(str(frozen.resolve()))
                if best is None or score > best:
                    best = score
                    best_history = str(frozen.resolve())
                    save(run_dir / "best.pt")
                history_paths = history_paths[-(cfg["history_capacity"] - 1) :]
                if best_history and best_history not in history_paths:
                    history_paths.insert(0, best_history)
                log(
                    "evaluation",
                    score=score,
                    w=w,
                    teacher=result["teacher"],
                    difference=result["teacher_minus_student"],
                    reference=result.get("reference_minus_student"),
                )
            if time.time() - last_save >= cfg["checkpoint_seconds"]:
                save(existing)
                last_save = time.time()
        save(existing)
        snapshot("final", max(0, deadline - started))
        if (
            bc_until > started
            and time.time() >= min(bc_until, deadline)
            and not (run_dir / "bc-final.pt").exists()
        ):
            save(run_dir / "bc-final.pt")
        atomic_json(
            run_dir / "status.json",
            {
                "state": "complete",
                "iteration": iteration,
                "deadline": deadline,
                "finished": time.time(),
                "best_score": best,
            },
        )
        return existing
    except BaseException as exc:
        atomic_json(
            run_dir / "status.json",
            {"state": "failed", "error": str(exc), "time": time.time()},
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--config")
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--initial")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--carry", action="store_true", help="continue a completed trial in a new phase"
    )
    p.add_argument("--updates")
    p.add_argument("--bc-seconds", type=float, default=0)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--snapshot-seconds", type=int)
    p.add_argument(
        "--reference", help="fixed Actor checkpoint added to every evaluation"
    )
    args = p.parse_args()
    cfg = configuration(
        **(json.loads(Path(args.config).read_text()) if args.config else {})
    )
    train(
        args.run,
        cfg,
        args.bundle,
        args.panel,
        deadline=args.deadline,
        initial=args.initial,
        resume=args.resume,
        carry=args.carry,
        updates=args.updates,
        bc_seconds=args.bc_seconds,
        eval_enabled=not args.no_eval,
        max_updates=args.max_updates,
        snapshot_seconds=args.snapshot_seconds,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
