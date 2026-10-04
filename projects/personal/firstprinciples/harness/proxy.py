"""Proxy tables: every version also plays whole games against the tournament crowd's stand-in.

The proxy panel imitates the platform players we meet (Actors cloned from
their own replays, ADR-0015); the long tail of rarer players is one more
proxy. Since 2026-10-05 (user decision) every evaluated version plays the
active panel after its development and confirmation sets, on the same host;
the result joins its record as ``proxy`` and shows on the lines page. It is
evidence beside the fixed pool, never a gate: keep, discard and a line's best
are decided on the pool alone.

  python -m harness.proxy register <actors dir> --job <job.json> --name "代理面板 v2" [--built 2026-10-05] [--check "..."]
  python -m harness.proxy activate <panel.json> [--reference fp-rule-131] [--name ...]   # maintainer; reseal afterwards
  python -m harness.proxy run <version> --host <host>                        # a version recorded before

Activation freezes the panel as a set of tables: one lineup of three proxies
per block, drawn by each player's share of our recent tables (a named proxy
sits at most once at a table, the tail may fill several seats), with the
fixed pool's runtime, game format and limits. Every version plays the same
games and is paired block by block with the reference version on them. The
reference is played once per host, because a model's near-tie argmax may
differ between inference backends; pairs are formed on one host only.
A new panel is a new activation; results stay with their panel id.
"""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

from . import evaluate as E

BLOCKS = E.override("FP_PROXY_BLOCKS", 1500)  # 4 games each: 6000 whole games
TAIL_SEATS = ("tail", "tail-2", "tail-3")


def panel_path():
    return E.LOCAL / "proxy" / "panel.json"


def results_path():
    return E.LOCAL / "proxy-results.json"


def load():
    """The active panel's tables, or None."""
    path = panel_path()
    return E.read_json(path) if path.exists() else None


def lineups(shares, blocks, seed):
    """One opponent lineup per block, drawn by recent table share."""
    rng = random.Random(seed)
    names = sorted(shares)
    weights = [shares[n] for n in names]
    out = []
    for _ in range(blocks):
        seats, tails = [], 0
        while len(seats) < 3:
            name = rng.choices(names, weights)[0]
            if name == "tail":
                seats.append(TAIL_SEATS[tails])
                tails += 1
            elif name not in seats:
                seats.append(name)
        out.append(seats)
    return out


def tables(panel, pool, reference, blocks=None):
    """The panel as harness tables (a pool whose block b has lineups[b] as opponents)."""
    blocks = blocks or BLOCKS
    members = {}
    for name, m in panel["members"].items():
        spec = E.with_timeouts({"family": "model", "version": name, "parameters": {"weights": m["weights"]}})
        for seat in (TAIL_SEATS if name == "tail" else (name,)):
            members[seat] = dict(spec, version=seat)
    shares = {n: m["share"] for n, m in panel["members"].items()}
    digest = hashlib.sha256(json.dumps([shares, members, blocks, pool["runtime_path"], pool["config"],
                                        pool["timeouts"]], sort_keys=True).encode()).hexdigest()
    seed = int(digest[:15], 16)
    # The panel id names the proxies (results of other table draws with them share it); the tables id names these games.
    return {"id": f"proxy-{digest[:16]}", "panel": panel_id(panel)[:8], "name": panel["name"], "members": members,
            "lineups": lineups(shares, blocks, seed), "blocks": blocks, "seed": seed,
            "runtime_path": pool["runtime_path"], "config": pool["config"], "timeouts": pool["timeouts"],
            "resources": pool["resources"], "reference": reference, "activated_at": time.time()}


def panel_id(panel):
    return panel.get("id") or hashlib.sha256(json.dumps(panel["members"], sort_keys=True).encode()).hexdigest()[:16]


def spec_of(version):
    rows = [r for r in E.attempts() if r["id"] == version and r.get("spec")]
    if not rows:
        raise E.Refused(f"没有 {version} 的评估记录")
    return rows[-1]["spec"]


def play(t, version, spec, host):
    scores, stats = E.play(t, version, spec, range(t["blocks"]), t["seed"], f"proxy-{host}", host)
    return scores, stats


def reference_blocks(t, host):
    """The reference version's blocks on this host (played once, then read)."""
    path = E.LOCAL / "proxy" / f"reference-{t['id']}-{host}.json"
    if path.exists():
        return {int(k): v for k, v in E.read_json(path)["blocks"].items()}
    scores, stats = play(t, t["reference"]["id"], t["reference"]["spec"], host)
    mean, se = E.mean_se(scores[b] for b in range(t["blocks"]))
    E.atomic_json(path, {"id": t["reference"]["id"], "host": host, "score": mean, "se": se, "stats": stats,
                         "blocks": {str(k): scores[k] for k in sorted(scores)}, "at": time.time()})
    append_result({"panel": t["panel"], "version": t["reference"]["id"], "score": mean, "se": se,
                   "games": t["blocks"] * 4, "seed": t["seed"], "host": host, "batch": "代理桌参照",
                   "at": time.time()})
    E.log(f"proxy reference {t['reference']['id']} on {host}: {mean:+.2f} ± {se:.2f}")
    return scores


def score(version, spec, host):
    """The record's ``proxy``: whole games on the active panel, paired with its reference on the same host."""
    t = load()
    if t is None:
        return None
    theirs = reference_blocks(t, host)
    if version == t["reference"]["id"]:
        mine, stats = theirs, None
    else:
        mine, stats = play(t, version, spec, host)
    blocks = range(t["blocks"])
    mean, se = E.mean_se(mine[b] for b in blocks)
    diff, dse = E.paired(mine, theirs, blocks) if version != t["reference"]["id"] else (0.0, 0.0)
    return {"panel": t["panel"], "games": t["blocks"] * 4, "score": mean, "se": se, "seed": t["seed"],
            "host": host, "batch": "默认代理桌评估", "stats": stats,
            "against": {"version": t["reference"]["id"], "diff": diff, "se": dse}}


def append_result(entry):
    import fcntl

    path = results_path()
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several hosts' evaluations may finish at once
        value = E.read_json(path) if path.exists() else {"panels": {}, "results": []}
        value["results"].append(entry)
        E.atomic_json(path, value)


def register(actors, job, name, built=None, check=None):
    """Copy a trained panel's Actors into the model store; write <actors>/../panel.json (not yet active)."""
    actors = Path(actors).resolve()
    job = E.read_json(Path(job))
    report = actors.parent / "report.json"
    members = {}
    for p in job["proxies"]:
        src = actors / p["name"]
        meta = E.read_json(src / "meta.json")
        sha = hashlib.sha256((src / "actor.pt").read_bytes()).hexdigest()
        if sha != meta["weights_sha256"]:
            raise E.Refused(f"{p['name']}：actor.pt 与 meta.json 的哈希不符")
        target = E.MODELS / sha
        if not target.exists():
            shutil.copytree(src, target)
        members[p["name"]] = {"weights": sha, "display": p["display"], "users": p["users"],
                              "share": p["recent_share"]}
    panel = {"name": name, "built": built, "check": check, "games": len(job.get("games") or []) or None,
             "members": members, "proxy_run": str(actors.parent),
             "proxy_config": E.read_json(report)["config"] if report.exists() else None}
    panel["id"] = panel_id(panel)
    out = actors.parent / "panel.json"
    E.atomic_json(out, panel)
    print(f"{len(members)} 个代理已登记 → {out}")


def activate(panel_file, reference, name=None):
    panel = E.read_json(Path(panel_file))
    panel["name"] = name or panel.get("name") or panel_id(panel)[:8]
    ref = {"id": reference, "spec": spec_of(reference)}
    t = tables(panel, E.load_pool(), ref)
    (E.LOCAL / "proxy").mkdir(parents=True, exist_ok=True)
    E.atomic_json(panel_path(), t)
    path = results_path()
    value = E.read_json(path) if path.exists() else {"panels": {}, "results": []}
    known = value.setdefault("panels", {}).setdefault(t["panel"], {})
    known.update({k: v for k, v in {"name": panel["name"], "built": panel.get("built"), "games": panel.get("games"),
                                    "proxies": len(panel["members"]), "check": panel.get("check")}.items()
                  if v is not None})
    E.atomic_json(path, value)
    print(f"代理面板 {t['panel']}（{panel['name']}）已启用：{t['blocks'] * 4} 场，参照 {reference}。"
          "重新 seal 后，新评估的版本默认加跑。")


def run(version, host):
    """Score a version recorded before the panel was active (it then shows on the lines page)."""
    E.check_seal()
    entry = score(version, spec_of(version), host)
    if entry is None:
        raise E.Refused("还没有启用的代理面板")
    if version != entry["against"]["version"]:
        append_result(dict(entry, version=version, batch="代理桌补测", at=time.time()))
    a = entry["against"]
    print(f"{version}：代理桌 {entry['score']:+.2f} ± {entry['se']:.2f}，对 {a['version']} {a['diff']:+.2f} ± {a['se']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("register")
    g.add_argument("actors")
    g.add_argument("--job", required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--built")
    g.add_argument("--check")
    a = sub.add_parser("activate")
    a.add_argument("panel")
    a.add_argument("--reference", default="fp-rule-131")
    a.add_argument("--name", help="shown on the lines page (default: the panel's)")
    r = sub.add_parser("run")
    r.add_argument("version")
    r.add_argument("--host", required=True, choices=tuple(E.EVAL_WORKERS))
    args = parser.parse_args()
    try:
        if args.command == "register":
            register(args.actors, args.job, args.name, args.built, args.check)
        elif args.command == "activate":
            activate(args.panel, args.reference, args.name)
        else:
            run(args.version, args.host)
    except E.Refused as exc:
        print(f"拒绝：{exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
