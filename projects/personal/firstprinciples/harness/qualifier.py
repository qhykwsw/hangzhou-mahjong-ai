"""Qualifier versions (海选): built to win more against the tournament crowd in the open round.

Since 2026-10-05 (user decision) the competition gets two versions. The two
iteration lines keep building the semifinal and final version against strong
opponents (the fixed pool, harness.evaluate). The third-party helpers build
the qualifier version, for an open round where most of the field is weaker:
it may expect to lose a little against strong opponents if it wins clearly
more against the crowd. There are two qualifier lines, q1 (led by Codex) and
q2 (led by the Claude helper), rule or RL alike. A candidate is judged
against its line's best version so far (before the first one, the proxy
panel's reference version):

1. the active proxy tables (the held-out half of the panel; candidates train
   and tune against the other half only): its paired difference must be
   above 0, or it stops here;
2. the development set: at most POOL_FLOOR per game below the better of the
   two lines' best versions (strong opponents), or it stops here;
3. fresh proxy walls, played by both: the one-sided 90% lower bound of the
   paired difference above 0 makes it the new best qualifier version.

The fixed tables and the development set are the same games for every
candidate; only the fresh walls of step 3 decide. Records go to
.local/qualifier.jsonl and their proxy-table scores to the lines page.

  python -m harness.qualifier submit <rule package dir | Actor dir | version> --line q1|q2 --host <host> \
      --name "..." [--why "..."]
  python -m harness.qualifier submit --resume fp-q1-003      # after an interruption
  python -m harness.qualifier list [--line q1|q2]
"""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

from . import evaluate as E
from . import proxy

POOL_FLOOR = 2.0  # per game below the lines' best on the development set (user: 「适当分数的期望下降」)
LINES = ("q1", "q2")


def records_path():
    return E.LOCAL / "qualifier.jsonl"


def records(line=None):
    path = records_path()
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    return [r for r in rows if line is None or r["line"] == line]


def champion(t, line):
    """The line's best qualifier version so far, or the proxy panel's reference before the first."""
    kept = [r for r in records(line) if r["status"] == "keep"]
    return {"id": kept[-1]["id"], "spec": kept[-1]["spec"]} if kept else t["reference"]


def strongest(pool):
    """The better of the two lines' best versions on the development set."""
    best = [r for r in (E.best(pool["id"], line) for line in ("rule", "rl")) if r]
    return max(best, key=lambda r: r["score"]) if best else None


def next_id(line):
    root, prefix = E.LOCAL / "qualifier", f"fp-{line}-"
    taken = [r["id"] for r in records()] + ([p.name for p in root.iterdir()] if root.exists() else [])
    numbers = [int(i[len(prefix):]) for i in taken if i.startswith(prefix) and i[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def freeze(source, qid):
    """The candidate's spec: a rule package copied under the id, an Actor in the model store, or a line version."""
    path = Path(source)
    if not path.exists():
        return dict(proxy.spec_of(source), version=qid)
    if (path / "actor.pt").is_file():
        meta = E.read_json(path / "meta.json")
        sha = hashlib.sha256((path / "actor.pt").read_bytes()).hexdigest()
        if sha != meta.get("weights_sha256"):
            raise E.Refused("actor.pt 与 meta.json 的哈希不符")
        if not (E.MODELS / sha).exists():
            shutil.copytree(path, E.MODELS / sha)
        return {"family": "model", "version": qid, "parameters": {"weights": sha}}
    problems = E.check_source(path)
    if problems:
        raise E.Refused("规则包不符合约定（进程、线程与资源归维护者）：\n" + "\n".join(problems))
    target = E.LOCAL / "qualifier" / qid / "package"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__"))
    error = E.build_native(target)
    if error:
        raise E.Refused("native.c 编译不过：\n" + error)
    return {"family": "code", "version": qid, "parameters": {}, "model_path": None, "program": None,
            "package_path": str(target), "package_hash": E.tree_hash(target),
            "python_path": str(E.REPO / ".venv/bin/python")}


def fresh(t, seed):
    """The same panel on new walls and newly drawn lineups."""
    return dict(t, id=f"{t['id']}-fresh-{seed:016x}", seed=seed, lineups=proxy.lineups(t["shares"], t["blocks"], seed))


def submit(source, line, host, name, why=None, resume=None):
    E.check_seal()
    t = proxy.load()
    if t is None:
        raise E.Refused("还没有启用的代理面板")
    pool = E.load_pool()
    state_path = E.LOCAL / "qualifier" / (resume or next_id(line)) / "state.json"
    if resume:
        state = E.read_json(state_path)
    else:
        qid = state_path.parent.name
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"id": qid, "line": line, "name": name, "why": why, "source": str(source), "host": host,
                 "spec": freeze(source, qid), "seed": random.SystemRandom().getrandbits(62),
                 "started_at": time.time()}
        E.atomic_json(state_path, state)
    qid, spec, host = state["id"], state["spec"], state["host"]
    best = champion(t, state["line"])
    record = {k: state[k] for k in ("id", "line", "name", "why", "source", "host", "spec", "started_at")}
    record["against"] = best["id"]
    blocks = range(t["blocks"])

    # 1. The fixed proxy tables, paired with the best qualifier version on this host.
    mine, stats = proxy.play(t, qid, spec, host)
    reference = proxy.reference_blocks(t, host)
    theirs = reference if best["id"] == t["reference"]["id"] else proxy.play(t, best["id"], best["spec"], host)[0]
    mean, se = E.mean_se(mine[b] for b in blocks)
    diff, dse = E.paired(mine, theirs, blocks)
    rd, rse = E.paired(mine, reference, blocks)
    record["proxy"] = {"panel": t["panel"], "tables": t["id"], "games": t["blocks"] * 4, "score": mean, "se": se,
                       "diff": diff, "diff_se": dse, "stats": stats}
    proxy.append_result({"panel": t["panel"], "version": qid, "score": mean, "se": se, "games": t["blocks"] * 4,
                         "seed": t["seed"], "host": host, "batch": "海选候选", "at": time.time(),
                         "against": {"version": t["reference"]["id"], "diff": rd, "se": rse}})
    E.log(f"{qid} proxy tables: {mean:+.2f} ± {se:.2f}, vs {best['id']} {diff:+.2f} ± {dse:.2f}")
    status = "discard" if diff <= 0 else None

    # 2. Strong opponents: the development set, against the lines' best.
    if status is None:
        top = strongest(pool)
        scores, stats = E.play(pool, qid, spec, range(E.DEV_BLOCKS), pool["dev_seed"], "dev", host)
        score, sse = E.mean_se(scores[b] for b in range(E.DEV_BLOCKS))
        record["pool"] = {"id": pool["id"], "games": E.DEV_BLOCKS * 4, "score": score, "se": sse, "stats": stats}
        if top:
            d, s = E.paired(scores, {int(k): v for k, v in top["dev_blocks"].items()}, range(E.DEV_BLOCKS))
            record["pool"].update(against=top["id"], diff=d, diff_se=s)
            E.log(f"{qid} dev: {score:+.2f}, vs {top['id']} {d:+.2f} ± {s:.2f}")
            status = "pool" if d < -POOL_FLOOR else None

    # 3. Fresh proxy walls, both versions: this decides.
    if status is None:
        f = fresh(t, state["seed"])
        mine, _ = E.play(f, qid, spec, blocks, f["seed"], "fresh", host)
        theirs, _ = E.play(f, best["id"], best["spec"], blocks, f["seed"], f"fresh-for-{qid}", host)
        d, s = E.paired(mine, theirs, blocks)
        record["confirm"] = {"seed": f["seed"], "games": t["blocks"] * 4, "diff": d, "se": s,
                             "lower90": d - E.CONFIRM_Z * s}
        status = "keep" if d - E.CONFIRM_Z * s > 0 else "unconfirmed"
    record.update(status=status, finished_at=time.time())
    with records_path().open("a") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    state_path.unlink()
    return record


STATUS = {"keep": "成功（新的海选版本）", "discard": "失败（代理桌不领先）", "pool": f"失败（对强手掉了 {POOL_FLOOR:g} 分以上）",
          "unconfirmed": "未确认（新牌墙上下界不大于 0）"}


def report(r):
    p = r["proxy"]
    lines = [f"{r['id']}（{r['name']}）：{STATUS[r['status']]}",
             f"代理桌 {p['games']} 场：{p['score']:+.2f} ± {p['se']:.2f}，对 {r['against']} {p['diff']:+.2f} ± {p['diff_se']:.2f}"]
    if r.get("pool"):
        q = r["pool"]
        lines.append(f"对手池开发集 {q['games']} 场：{q['score']:+.2f} ± {q['se']:.2f}"
                     + (f"，对 {q['against']} {q['diff']:+.2f} ± {q['diff_se']:.2f}" if q.get("against") else ""))
    if r.get("confirm"):
        c = r["confirm"]
        lines.append(f"新牌墙代理桌 {c['games']} 场：对 {r['against']} {c['diff']:+.2f} ± {c['se']:.2f}，"
                     f"90% 下界 {c['lower90']:+.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("source", nargs="?", help="规则包目录、Actor 目录（actor.pt + meta.json），或已评估的版本 id")
    s.add_argument("--line", choices=LINES, help="海选 1（Codex 主导）或海选 2（Claude 帮手主导）")
    s.add_argument("--host", choices=tuple(E.EVAL_WORKERS))
    s.add_argument("--name", default="")
    s.add_argument("--why")
    s.add_argument("--resume", help="中断后续跑这个 id（沿用它的冻结与新牌墙种子）")
    sub.add_parser("list").add_argument("--line", choices=LINES)
    args = parser.parse_args()
    try:
        if args.command == "list":
            for r in records(args.line):
                print(report(r), end="\n\n")
        else:
            if not args.resume and not (args.source and args.line and args.host):
                raise E.Refused("要给出候选、--line 与 --host，或用 --resume 续跑")
            print(report(submit(args.source, args.line, args.host, args.name, args.why, args.resume)))
    except E.Refused as exc:
        print(f"拒绝：{exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
