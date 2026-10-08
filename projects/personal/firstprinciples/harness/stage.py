"""Simulated tournament stages: does a version advance more often (user, 10-06)?

The semifinal and final are stages of 10 whole games at one table of four,
ranked on the stage's total score (the semifinal advances its top two, the
final crowns one; the final breaks score ties with extra games). Here a stage
is one match of the fixed pool's game format with ``games_per_round`` 10: the
version against three distinct pool members, the stage's scores, place and
remaining games in every observation (StageContext), seats rotating between
games. Block b is one stage: lineup b mod 35, the version's starting seat
taking the four seats in turn. Every version plays the same stages, paired
block by block with a reference version (by default its line's best).

  python -m harness.stage run <version> --kind semi|final --host <host> [--against <version>] [--stages N]
  python -m harness.stage run <name> --spec <policy.json> --kind semi|final --host <host> --against <version> [--stages N]
  python -m harness.stage run <name> --weights <model id> --against <version> --kind ... --host ...
  python -m harness.stage list

``--weights`` plays a model of the research model store by its id under
``name`` without an evaluation record, e.g. a version with the stage re-rank
switched on (``python -m guided.outcome variant <its artifact> <model store>``
prints the id), paired with ``--against`` (e.g. the same version without it).

A record holds the advance rate (semifinal: top two, a tie for second
counting half; final: first), the paired difference with the reference and
its one-sided 90% lower bound, and the stage scores per game. Results are on
record for the user's version choice; the lines' development and
confirmation verdicts are unchanged. Each version plays once per host (as on
the proxy tables, a model's near-tie argmax may differ between backends), and
pairs are formed on one host only.

A model policy that is no recorded version (--spec, e.g. one that switches
between two models by the stage's score, model_seat.Switch) plays under a
name of its own and pairs with the version given. Its jobs are keyed by the
spec and the runtime it plays on, so they are never another spec's. A seat
with a switch is newer than the pool's runtime: it plays on the pool's
runtime with the repository's model seat files (seat_runtime), where the
other three seats, and a seat without a switch, play exactly as on the pool's
own, so it pairs with versions played there. Its record keeps the spec and
the share of its decisions made by the second model.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import evaluate as E
from .proxy import spec_of

STAGES = E.override("FP_STAGE_BLOCKS", 2000)
GAMES = 10
KINDS = {"semi": {"stage": "advancement", "advance": 2, "name": "半决赛（前二晋级）"},
         "final": {"stage": "final", "advance": 1, "name": "决赛（第一）"}}
TIEBREAK_HANDS = 64  # an execution budget for the final's tiebreak games, not a rule
# The model seat as the repository has it (src/mahjong_ai): the switch, its
# validation in PolicySpec, and its count in a job's readout.
SEAT_FILES = ("execution/model_seat.py", "strategies/common.py", "evaluation/schedule.py")


def tables(kind):
    """The fixed pool as stage tables of ``kind``: its own id, game format and seed."""
    pool = E.load_pool()
    config = dict(pool["config"], games_per_round=GAMES, stage=KINDS[kind]["stage"])
    if kind == "final":
        config["max_tiebreak_rounds"] = TIEBREAK_HANDS
    seed = int(hashlib.sha256(f"{pool['dev_seed']}-stage-{kind}".encode()).hexdigest()[:15], 16)
    return dict(pool, id=f"{pool['id']}-stage-{kind}", config=config, one_seat=True, stage_seed=seed)


def advanced(place, advance):
    """1 when the place advances, a half for a tie straddling the line (the final's midranks)."""
    return 1.0 if place <= advance else 0.5 if place - advance == 0.5 else 0.0


def play(t, version, spec, host, stages, key=""):
    # The job split depends on the number of stages: a run of another length must not reuse these jobs.
    scores, stats = E.play(t, version, spec, range(stages), t["stage_seed"], f"stage{stages}-{host}{key}", host)
    return scores, stats.pop("places"), stats


def seat_runtime(pool):
    """The pool's runtime with the repository's SEAT_FILES, stored under its fingerprint like every runtime."""
    old, new = Path(pool["runtime_path"]) / "mahjong_ai", E.REPO / "src" / "mahjong_ai"
    if all((old / f).read_bytes() == (new / f).read_bytes() for f in SEAT_FILES):
        return pool["runtime_path"]
    store = E.RESEARCH / ".local/research/runtimes"
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=store, prefix=".seat-") as tmp:
        # _fast.c is the pool's, so its compiled accelerator comes along.
        shutil.copytree(old, Path(tmp) / "mahjong_ai", ignore=shutil.ignore_patterns("__pycache__"))
        for f in SEAT_FILES:
            shutil.copyfile(new / f, Path(tmp) / "mahjong_ai" / f)
        fingerprint = subprocess.run(
            [sys.executable, "-c", "from mahjong_ai.persistence.artifacts import runtime_fingerprint;print(runtime_fingerprint())"],
            env=dict(os.environ, PYTHONPATH=tmp, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, check=True).stdout.strip()
        target = store / fingerprint
        if not target.exists():
            shutil.move(tmp, target)
            Path(tmp).mkdir()  # the context manager removes it
    return str(target)


def custom(name, spec, t):
    """A model policy from a spec file, played as ``name``: (its spec, its tables, its job key)."""
    from mahjong_ai.execution.model_seat import served_weights
    from mahjong_ai.strategies.common import PolicySpec

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise E.Refused(f"名字 {name!r} 只能用字母、数字和 ._-")
    if name in t["members"] or any(r["id"] == name for r in E.attempts()):
        raise E.Refused(f"{name} 已是版本或对手池成员的编号，--spec 的策略要另起名字")
    spec = dict(spec, version=name)
    try:
        if PolicySpec(**spec).family != "model":
            raise ValueError("只接受模型策略")
    except (TypeError, ValueError) as exc:
        raise E.Refused(f"--spec 不是可用的策略：{exc}") from exc
    missing = [w[:12] for w in sorted(served_weights(spec)) if not (E.MODELS / w).is_dir()]
    if missing:
        raise E.Refused(f"模型库里没有 {', '.join(missing)}")
    if spec["parameters"].get("switch"):
        t = dict(t, runtime_path=seat_runtime(t))
    key = hashlib.sha256(json.dumps([spec, Path(t["runtime_path"]).name], sort_keys=True).encode()).hexdigest()
    return spec, t, f"-spec{key[:12]}"


def results_path():
    return E.LOCAL / "stage-results.jsonl"


def run(version, kind, host, against=None, stages=STAGES, spec=None):
    """Play ``version`` (or, with ``spec``, that policy under the name ``version``) and record it."""
    t = tables(kind)
    E.check_seal()
    if spec is not None:
        if against is None:
            raise E.Refused("用 --spec 时要用 --against 指定配对的版本")
        mine_spec, mine_t, key = custom(version, spec, t)
    else:
        if against is None:
            line = next(r["line"] for r in E.attempts() if r["id"] == version)
            against = E.best(E.load_pool()["id"], line)["id"]
        mine_spec, mine_t, key = spec_of(version), t, ""
    blocks = range(stages)
    advance = KINDS[kind]["advance"]
    mine_scores, mine_places, stats = play(mine_t, version, mine_spec, host, stages, key)
    theirs_scores, theirs_places, _ = play(t, against, spec_of(against), host, stages)
    mine = {b: advanced(mine_places[b], advance) for b in blocks}
    theirs = {b: advanced(theirs_places[b], advance) for b in blocks}
    rate, rate_se = E.mean_se(mine[b] for b in blocks)
    diff, se = E.paired(mine, theirs, blocks)
    score_diff, score_se = E.paired(mine_scores, theirs_scores, blocks)
    out = {"version": version, "kind": kind, "against": against, "host": host, "stages": stages,
           "games": stages * GAMES, "seed": t["stage_seed"], "pool": E.load_pool()["id"],
           "rate": rate, "rate_se": rate_se, "against_rate": sum(theirs.values()) / stages,
           "diff": diff, "se": se, "lower90": diff - E.CONFIRM_Z * se,
           "score_per_game": E.mean_se(mine_scores[b] for b in blocks)[0] / GAMES,
           "score_diff_per_game": score_diff / GAMES, "score_se_per_game": score_se / GAMES,
           "stats": stats, "at": time.time()}
    if spec is not None:
        out.update(spec=mine_spec, runtime=Path(mine_t["runtime_path"]).name)
        if "alt_actions" in stats:
            out["alt_share"] = stats["alt_actions"] / max(1, stats["actions"])
    with results_path().open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def report(r):
    k = KINDS[r["kind"]]
    return (f"{r['version']} {k['name']}，{r['stages']} 个阶段（{r['games']} 场，{r['host']}）：晋级率 {r['rate']:.1%}"
            f"（{r['against']} {r['against_rate']:.1%}），配对差 {r['diff'] * 100:+.2f}±{r['se'] * 100:.2f} 个百分点，"
            f"下界 {r['lower90'] * 100:+.2f}；场均分 {r['score_per_game']:+.2f}（差 {r['score_diff_per_game']:+.2f}"
            f"±{r['score_se_per_game']:.2f}）" + (f"；第二个模型做了 {r['alt_share']:.1%} 的决策" if "alt_share" in r else ""))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("version")
    r.add_argument("--kind", required=True, choices=sorted(KINDS))
    r.add_argument("--host", required=True)
    r.add_argument("--against")
    r.add_argument("--stages", type=int, default=STAGES)
    r.add_argument("--spec", type=Path, help="a model policy (JSON) to play under the name given as version")
    r.add_argument("--weights", help="play this model id of the model store as <version> (needs --against)")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    if args.command == "run":
        spec = json.loads(args.spec.read_text()) if args.spec else None
        if args.weights:  # a model of the store by its id, e.g. a re-rank variant (guided.outcome)
            if spec is not None:
                raise SystemExit("--spec 和 --weights 只能用一个")
            spec = {"family": "model", "version": args.version, "parameters": {"weights": args.weights}}
        try:
            print(report(run(args.version, args.kind, args.host, args.against, args.stages, spec)))
        except E.Refused as exc:
            print(f"拒绝：{exc}")
            raise SystemExit(2)
    elif results_path().exists():
        for line in results_path().read_text().splitlines():
            print(report(json.loads(line)))


if __name__ == "__main__":
    main()
