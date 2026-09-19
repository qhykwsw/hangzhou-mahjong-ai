"""Evaluate every saved node of a training run, optionally after the run exits.

    PYTHONPATH=src python3 projects/personal/selfplay/train/sweep_nodes.py --run projects/personal/selfplay/runs/transformer-main --wait-pid 12345

Nodes are the half-hourly league snapshots, ckpt/best*.pt and the final export.
Stage 1 plays every node against one frozen opponent set with one seed, so all
nodes meet the same tables and walls and are compared per block against the
anchor.  Stage 2 puts the top nodes, the final export and the anchor into a brawl
with the same fixed opponents under a new seed, which removes the selection bias
of picking stage-1 maxima.  The opponents are the run's own training opponents:
its opponent_entries list when set, else the built-in baselines plus the
auto-research representatives.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
for sub in (HERE, PROJECT / "src", REPO / "src"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from config import load  # noqa: E402
from opponents import default_entries  # noqa: E402


def log(message):
    print(f"[sweep {time.strftime('%H:%M:%S')}] {message}", flush=True)


def wait_for(pid):
    log(f"waiting for pid {pid}")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(60)


def load_cfg(run_dir):
    cfg = load(run_dir / "config.json")
    cfg.device = "cpu"
    return cfg


def collect_nodes(run_dir):
    import torch

    rows = [json.loads(line) for line in (run_dir / "log.jsonl").open()]
    rows = [r for r in rows if "iteration" in r and "time" in r]
    final_export = run_dir / "export" / "policy.onnx"
    nodes = []
    for path in sorted((run_dir / "league").glob("ckpt-*.pt")):
        stamp = path.stat().st_mtime
        # A snapshot is written just before the log row of its iteration.
        row = next((r for r in rows if r["time"] >= stamp), None)
        if row is None or (row.get("final") and final_export.exists()):
            continue  # the closing snapshot has the same weights as the final export
        nodes.append({"name": path.stem, "source": path, "iteration": row["iteration"],
                      "hours": row["elapsed_min"] / 60, "saved": stamp})
    for path in sorted((run_dir / "ckpt").glob("best*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if any(n["iteration"] == state["iteration"] and n["name"].startswith("best") for n in nodes):
            continue
        nodes.append({"name": path.stem, "source": path, "iteration": state["iteration"],
                      "hours": state["elapsed"] / 3600, "saved": path.stat().st_mtime})
    final = next((r for r in reversed(rows) if r.get("final")), None)
    if final and final_export.exists():
        nodes.append({"name": "final", "export": final_export.parent, "iteration": final["iteration"],
                      "hours": final["elapsed_min"] / 60, "saved": final_export.stat().st_mtime})
    return sorted(nodes, key=lambda n: (n["iteration"], n["name"]))


def export_node(node, cfg, out_root):
    if "export" in node:
        return node["export"]
    out = out_root / "exports" / f"{node['name']}-it{node['iteration']}"
    if (out / "policy.onnx").exists():
        return out
    import torch

    from export import export_onnx, verify_onnx
    from models import build_model

    state = torch.load(node["source"], map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    export_onnx(model, cfg, out, version=f"{cfg.run_name}-{node['name']}-it{node['iteration']}")
    parity = verify_onnx(model, out, cfg)
    if not parity["ok"]:
        raise RuntimeError(f"ONNX parity failed for {node['name']}: {parity}")
    return out


def run_evaluate(extra, output, entries, workers):
    if not output.exists():
        cmd = [sys.executable, str(HERE / "evaluate.py"), *extra, "--entries", str(entries), "--output", str(output)]
        if workers:
            cmd += ["--workers", str(workers)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text())


def block_means(records, spec):
    blocks = {}
    for r in records:
        blocks.setdefault(r["seed"] // 4, []).append(r["scores"][r["seats"].index(spec)])
    return {b: float(np.mean(v)) for b, v in blocks.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(PROJECT / "runs" / "transformer-main"))
    p.add_argument("--out", help="default: <run>/sweep")
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--anchor", default="best-it45")
    p.add_argument("--only", nargs="*", help="restrict to these node names")
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--brawl-games", type=int, default=2048)
    p.add_argument("--brawl-seed", type=int, default=7)
    p.add_argument("--workers", type=int)
    args = p.parse_args()

    if args.wait_pid:
        wait_for(args.wait_pid)
    run_dir = Path(args.run).resolve()
    out_root = Path(args.out).resolve() if args.out else run_dir / "sweep"
    (out_root / "stage1").mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(run_dir)
    entries = out_root / "opponents.json"
    if not entries.exists():
        # 与训练用同一份固定对手：run 设了 opponent_entries 就只用它，否则是内置基线加研究代表。
        entries.write_text(json.dumps(default_entries(cfg, REPO), ensure_ascii=False, indent=1))
    opponent_ids = [e["id"] for e in json.loads(entries.read_text())]
    log(f"opponents: {opponent_ids}")

    nodes = collect_nodes(run_dir)
    if args.only:
        nodes = [n for n in nodes if n["name"] in args.only]
    if not any(n["name"] == args.anchor for n in nodes):
        raise SystemExit(f"anchor {args.anchor} not among nodes {[n['name'] for n in nodes]}")
    log(f"{len(nodes)} nodes: {[n['name'] for n in nodes]}")

    for node in nodes:
        node["dir"] = str(Path(export_node(node, cfg, out_root)).resolve())
        started = time.time()
        result = run_evaluate(["--policy", node["dir"], "--games", str(args.games), "--seed", str(args.seed)],
                              out_root / "stage1" / f"{node['name']}-it{node['iteration']}.json", entries, args.workers)
        node["summary"] = result["summary"]["results"][0]
        node["blocks"] = block_means(result["records"], node["dir"])
        log(f"{node['name']} it{node['iteration']}: {node['summary']['mean_score']:+.2f} "
            f"± {node['summary']['score_se']:.2f} ({time.time() - started:.0f}s)")

    anchor = next(n for n in nodes if n["name"] == args.anchor)
    for node in nodes:
        common = sorted(set(node["blocks"]) & set(anchor["blocks"]))
        diffs = np.asarray([node["blocks"][b] - anchor["blocks"][b] for b in common])
        node["diff"] = float(diffs.mean())
        node["diff_se"] = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")

    lines = [
        f"# {run_dir.name} 节点评估",
        "",
        f"阶段一：每个节点 {args.games} 场，seed {args.seed}，16 局，有财必拷响关闭；对手固定为 `opponents.json`，"
        f"配对差以 4 场一块计算标准误。",
        "",
        f"| 节点 | 保存时间 | 轮数 | 训练时长 h | 场均桌面分 | 相对 {args.anchor}（配对） | 平均名次 | 首名率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in nodes:
        s = n["summary"]
        lines.append(
            f"| {n['name']} | {time.strftime('%m-%d %H:%M', time.localtime(n['saved']))} | {n['iteration']} | "
            f"{n['hours']:.2f} | {s['mean_score']:+.2f} ± {s['score_se']:.2f} | {n['diff']:+.2f} ± {n['diff_se']:.2f} | "
            f"{s['mean_rank']:.2f} | {s['first_rate']:.3f} |"
        )

    if args.brawl_games > 0:
        ranked = sorted((n for n in nodes if n["name"] != args.anchor), key=lambda n: -n["diff"])
        picks = []
        for n in ranked[: args.top] + [n for n in nodes if n["name"] in ("final", args.anchor)]:
            if all(n["name"] != q["name"] for q in picks):
                picks.append(n)
        log(f"brawl with {[n['name'] for n in picks]}")
        brawl = run_evaluate(
            ["--policy", *[n["dir"] for n in picks], "--brawl", *opponent_ids,
             "--games", str(args.brawl_games), "--seed", str(args.brawl_seed)],
            out_root / "brawl.json", entries, args.workers,
        )
        names = {n["dir"]: n["name"] for n in picks}
        lines += ["", f"阶段二：大乱斗 {args.brawl_games} 场，seed {args.brawl_seed}", "",
                  "| 名次 | 参赛者 | 场均桌面分 | 名次分 | 首名率 | 场数 |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(brawl["summary"]["results"], 1):
            label = names.get(r["policy"], r["policy"])
            lines.append(f"| {i} | {label} | {r['mean_score']:+.2f} ± {r['score_se']:.2f} | "
                         f"{r['place_points']:+.2f} | {r['first_rate']:.3f} | {r['games']} |")

    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    log(f"done: {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
