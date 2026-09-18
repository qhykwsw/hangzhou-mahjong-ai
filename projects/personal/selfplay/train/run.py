"""Self-play PPO trainer entrypoint.

    .venv/bin/python projects/personal/selfplay/train/run.py --config configs/resnet.toml --hours 1

Workers are forked before torch/CUDA is initialised; the parent process owns the
GPU model, batched inference, PPO updates, the league and all logging.
"""

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for sub in (HERE, PROJECT / "src"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

import numpy as np  # noqa: E402

from config import REPO, load  # noqa: E402
from actors import worker_main  # noqa: E402
from opponents import default_entries  # noqa: E402
from rollout import FIELDS, REQUEST_FIELDS, SharedBlock  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--arch")
    p.add_argument("--run-name")
    p.add_argument("--hours", type=float)
    p.add_argument("--train-workers", type=int)
    p.add_argument("--eval-workers", type=int)
    p.add_argument("--tables", type=int)
    p.add_argument("--samples", type=int, help="closed transitions per worker per iteration")
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-auto-research", action="store_true")
    p.add_argument("--fresh", action="store_true", help="delete an existing run directory first")
    return p.parse_args()


def main():
    args = parse()
    cfg = load(
        args.config,
        arch=args.arch,
        run_name=args.run_name,
        hours=args.hours,
        train_workers=args.train_workers,
        eval_workers=args.eval_workers,
        tables_per_worker=args.tables,
        samples_per_worker=args.samples,
        seed=args.seed,
    )
    if args.no_auto_research:
        cfg.include_auto_research = False
    run_dir = cfg.run_dir
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=1, default=str))
    log_file = (run_dir / "log.jsonl").open("a")

    def log(row):
        row["time"] = time.time()
        log_file.write(json.dumps(row, default=str) + "\n")
        log_file.flush()

    entries = default_entries(cfg, REPO)
    print(f"[run] {cfg.run_name} arch={cfg.arch} fixed opponents: {[e['id'] for e in entries]}", flush=True)

    P_train, P_eval, T = cfg.train_workers, cfg.eval_workers, cfg.tables_per_worker
    P = P_train + P_eval
    capacity = cfg.samples_per_worker + 8 * T + 64
    req = SharedBlock(REQUEST_FIELDS, (P, T), create=True)
    store = SharedBlock(FIELDS, (P_train, capacity), create=True)
    layout = {
        "req_lead": (P, T),
        "req_names": req.names,
        "store_lead": (P_train, capacity),
        "store_names": store.names,
        "capacity": capacity,
    }
    ctx = mp.get_context("fork")
    conns, procs, roles = [], [], []
    base_seed = cfg.seed * 1000003
    for w in range(P):
        role = "train" if w < P_train else "eval"
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=worker_main,
            args=(w, role, cfg.as_dict(), layout, child_conn, entries, base_seed + w),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        conns.append(parent_conn)
        procs.append(proc)
        roles.append(role)
    train_conns = conns[:P_train]
    conn_index = {c: w for w, c in enumerate(conns)}

    # ---- GPU side (after fork) ------------------------------------------------
    import torch

    from models import TwoHot, build_model
    from league import League
    from learner import Batch, ppo_update
    from server import InferenceServer
    from export import export_onnx, verify_onnx

    torch.set_num_threads(cfg.threads)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] model {cfg.arch}: {n_params/1e6:.2f}M parameters on {device}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5, fused=(device.type == "cuda"))
    twohot = TwoHot(cfg.value_bins, cfg.value_range, device)
    server = InferenceServer(req, device, twohot, cfg)
    league = League(cfg, [e["id"] for e in entries], run_dir, device)
    rng = np.random.default_rng(cfg.seed)

    iteration, elapsed_offset, eval_history = 0, 0.0, []
    latest = run_dir / "ckpt" / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        iteration, elapsed_offset = state["iteration"], state["elapsed"]
        eval_history = state.get("eval_history", [])
        league.load_state_dict(state["league"], lambda: build_model(cfg))
        print(f"[run] resumed at iteration {iteration}, {elapsed_offset/3600:.2f}h", flush=True)
    model.eval()

    started = time.time()
    budget = cfg.hours * 3600

    def elapsed():
        return elapsed_offset + time.time() - started

    def frac():
        return min(1.0, elapsed() / max(1.0, budget))

    def save_checkpoint(tag="latest"):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "elapsed": elapsed(),
                "league": league.state_dict(),
                "eval_history": eval_history[-5000:],
                "config": cfg.as_dict(),
            },
            run_dir / "ckpt" / f"{tag}.pt",
        )

    def eval_summary():
        recent = eval_history[-cfg.eval_window:]
        if not recent:
            return {"eval_games": 0}
        scores = np.asarray([r["score"] for r in recent], dtype=np.float64)
        ranks = np.asarray([r["rank"] for r in recent], dtype=np.float64)
        return {
            "eval_games": len(recent),
            "eval_score": float(scores.mean()),
            "eval_score_se": float(scores.std() / max(1, np.sqrt(len(scores)))),
            "eval_rank": float(ranks.mean()),
            "eval_place_points": float((5 - 2 * ranks).mean()),
            "eval_total_games": len(eval_history),
        }

    def handle_records(worker, records):
        if roles[worker] == "train":
            league.record_games(records)
            return
        for rec in records:
            me = next(i for i, (k, ident) in enumerate(rec["seats"]) if k == "model")
            scores = rec["scores"]
            rank = 1 + sum(v > scores[me] for v in scores) + (sum(v == scores[me] for v in scores) - 1) / 2
            eval_history.append(
                {
                    "iteration": iteration,
                    "elapsed": elapsed(),
                    "score": scores[me],
                    "rank": rank,
                    "ycbk": rec["you_cai_bi_kao"],
                    "opponents": [ident for i, (k, ident) in enumerate(rec["seats"]) if i != me],
                }
            )

    # eval workers run continuously from the first snapshot on
    snap = league.snapshot(elapsed() / 3600)
    for c in conns[P_train:]:
        c.send(("collect", 0, snap, 0.0))

    last_ckpt = time.time()
    best_eval = -1e9
    try:
        while elapsed() < budget:
            it_start = time.time()
            iteration += 1
            f = frac()
            lr = cfg.learning_rate * (1 - (1 - cfg.lr_final_fraction) * f)
            for g in optimizer.param_groups:
                g["lr"] = lr
            entropy_coef = cfg.entropy_coef + (cfg.entropy_final - cfg.entropy_coef) * f
            shaping = cfg.shaping_weight * max(0.0, 1 - f / max(1e-6, cfg.shaping_anneal_fraction))
            snap = league.snapshot(elapsed() / 3600)
            for c in train_conns:
                c.send(("collect", cfg.samples_per_worker, snap, shaping))
            done = {}
            collect_times = []
            waiting = set(conns)  # connections that may still send us something

            def drain(ready, batch_ready):
                for c in ready:
                    try:
                        msg = c.recv()
                    except EOFError:
                        raise RuntimeError(f"worker {conn_index[c]} died")
                    w = conn_index[c]
                    kind = msg[0]
                    if kind == "req":
                        batch_ready.append((w, msg[1]))
                        waiting.discard(c)
                        if msg[2]:
                            handle_records(w, msg[2])
                    elif kind == "done":
                        done[w] = msg[1]
                        collect_times.append(msg[2])
                        waiting.discard(c)
                        if msg[3]:
                            handle_records(w, msg[3])
                    elif kind == "error":
                        raise RuntimeError(f"worker {msg[1]} failed:\n{msg[2]}")
                    else:
                        raise RuntimeError(f"unexpected message {kind} from worker {w}")

            while len(done) < P_train:
                batch_ready = []
                drain(wait(list(waiting), timeout=0.2), batch_ready)
                # Accumulate more workers for a few milliseconds so each GPU
                # forward sees a large batch instead of one worker's tables.
                active = P - len(done)
                deadline = time.time() + cfg.batch_wait_ms / 1000
                while batch_ready and len(batch_ready) < max(1, int(active * cfg.batch_target_fraction)) and waiting:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    drain(wait(list(waiting), timeout=remaining), batch_ready)
                if batch_ready:
                    server.serve(batch_ready, model, league.opponent_models())
                    for w, _ in batch_ready:
                        conns[w].send(("act",))
                        waiting.add(conns[w])
            collect_seconds = time.time() - it_start
            counts = [done[w] for w in range(P_train)]
            batch = Batch(store, counts, device)
            learn_start = time.time()
            stats = ppo_update(model, optimizer, batch, cfg, twohot, cfg.clip, entropy_coef, rng)
            learn_seconds = time.time() - learn_start
            snapshot_slot = league.maybe_snapshot(model)
            summary = eval_summary()
            row = {
                "iteration": iteration,
                "elapsed_min": round(elapsed() / 60, 2),
                "samples": batch.n,
                "env_sps": round(batch.n / max(1e-6, collect_seconds)),
                "collect_s": round(collect_seconds, 2),
                "learn_s": round(learn_seconds, 2),
                "lr": lr,
                "entropy_coef": entropy_coef,
                "shaping": round(shaping, 4),
                "fixed_prob": round(snap["fixed_prob"], 3),
                "league_size": len(league.checkpoints),
                "forward_batch": round(server.forward_rows / max(1, server.forward_calls), 1),
                "forward_calls": server.forward_calls,
                "server_forward_s": round(server.t_forward, 2),
                "server_write_s": round(server.t_write, 2),
                **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in stats.items()},
                **summary,
            }
            if snapshot_slot is not None:
                row["league_snapshot"] = snapshot_slot
                row["league"] = league.summary()
            log(row)
            print(
                f"[it {iteration:4d} {row['elapsed_min']:6.1f}m] n={batch.n} sps={row['env_sps']} "
                f"batch={row['forward_batch']} fwd={row['server_forward_s']:.1f}s learn={learn_seconds:.1f}s loss={stats['loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} ev={stats['ev_symlog']:.3f} "
                f"R={stats['reward_mean']:+.3f} eval={summary.get('eval_score', float('nan')):+.2f}"
                f"±{summary.get('eval_score_se', float('nan')):.2f} (n={summary.get('eval_games', 0)}) "
                f"league={len(league.checkpoints)} shaping={shaping:.2f}",
                flush=True,
            )
            server.reset_stats()
            if time.time() - last_ckpt > cfg.checkpoint_minutes * 60:
                save_checkpoint("latest")
                last_ckpt = time.time()
            # 窗口攒满才比较，免得早期零星几百场的高点占住 best。
            if summary.get("eval_games", 0) >= cfg.eval_window and summary["eval_score"] > best_eval:
                best_eval = summary["eval_score"]
                save_checkpoint("best")
        save_checkpoint("latest")
        league.maybe_snapshot(model, force=True)
        export_dir = run_dir / "export"
        export_onnx(model, cfg, export_dir, version=f"{cfg.run_name}-it{iteration}")
        parity = verify_onnx(model, export_dir, cfg)
        final = {"final": True, "iteration": iteration, "elapsed_min": round(elapsed() / 60, 2), "parity": parity, **eval_summary(), "league": league.summary()}
        log(final)
        print(f"[run] finished: {json.dumps({k: v for k, v in final.items() if k != 'league'}, default=str)}", flush=True)
    finally:
        for c in conns:
            try:
                c.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        req.close(unlink=True)
        store.close(unlink=True)
        log_file.close()


if __name__ == "__main__":
    main()
