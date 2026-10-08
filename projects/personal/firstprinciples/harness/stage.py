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
"""
import argparse
import hashlib
import json
import time

from . import evaluate as E
from .proxy import spec_of

STAGES = E.override("FP_STAGE_BLOCKS", 2000)
GAMES = 10
KINDS = {"semi": {"stage": "advancement", "advance": 2, "name": "半决赛（前二晋级）"},
         "final": {"stage": "final", "advance": 1, "name": "决赛（第一）"}}
TIEBREAK_HANDS = 64  # an execution budget for the final's tiebreak games, not a rule


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


def play(t, version, spec, host, stages):
    # The job split depends on the number of stages: a run of another length must not reuse these jobs.
    scores, stats = E.play(t, version, spec, range(stages), t["stage_seed"], f"stage{stages}-{host}", host)
    return scores, stats.pop("places"), stats


def results_path():
    return E.LOCAL / "stage-results.jsonl"


def run(version, kind, host, against=None, stages=STAGES, weights=None):
    t = tables(kind)
    E.check_seal()
    if weights is not None:
        if against is None:
            raise E.Refused("--weights 需要 --against 指定配对的版本")
        if not (E.MODELS / weights / "meta.json").is_file():
            raise E.Refused(f"模型库里没有 {weights}")
        spec = {"family": "model", "version": version, "parameters": {"weights": weights}}
    else:
        spec = spec_of(version)
    if against is None:
        line = next(r["line"] for r in E.attempts() if r["id"] == version)
        against = E.best(E.load_pool()["id"], line)["id"]
    blocks = range(stages)
    advance = KINDS[kind]["advance"]
    mine_scores, mine_places, stats = play(t, version, spec, host, stages)
    theirs_scores, theirs_places, _ = play(t, against, spec_of(against), host, stages)
    mine = {b: advanced(mine_places[b], advance) for b in blocks}
    theirs = {b: advanced(theirs_places[b], advance) for b in blocks}
    rate, rate_se = E.mean_se(mine[b] for b in blocks)
    diff, se = E.paired(mine, theirs, blocks)
    score_diff, score_se = E.paired(mine_scores, theirs_scores, blocks)
    out = {"version": version, "kind": kind, "against": against, "host": host, "stages": stages,
           **({"weights": weights} if weights is not None else {}),
           "games": stages * GAMES, "seed": t["stage_seed"], "pool": E.load_pool()["id"],
           "rate": rate, "rate_se": rate_se, "against_rate": sum(theirs.values()) / stages,
           "diff": diff, "se": se, "lower90": diff - E.CONFIRM_Z * se,
           "score_per_game": E.mean_se(mine_scores[b] for b in blocks)[0] / GAMES,
           "score_diff_per_game": score_diff / GAMES, "score_se_per_game": score_se / GAMES,
           "stats": stats, "at": time.time()}
    with results_path().open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def report(r):
    k = KINDS[r["kind"]]
    return (f"{r['version']} {k['name']}，{r['stages']} 个阶段（{r['games']} 场，{r['host']}）：晋级率 {r['rate']:.1%}"
            f"（{r['against']} {r['against_rate']:.1%}），配对差 {r['diff'] * 100:+.2f}±{r['se'] * 100:.2f} 个百分点，"
            f"下界 {r['lower90'] * 100:+.2f}；场均分 {r['score_per_game']:+.2f}（差 {r['score_diff_per_game']:+.2f}"
            f"±{r['score_se_per_game']:.2f}）")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("version")
    r.add_argument("--kind", required=True, choices=sorted(KINDS))
    r.add_argument("--host", required=True)
    r.add_argument("--against")
    r.add_argument("--stages", type=int, default=STAGES)
    r.add_argument("--weights", help="play this model id of the model store as <version> (needs --against)")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.command == "run":
        try:
            print(report(run(args.version, args.kind, args.host, args.against, args.stages, args.weights)))
        except E.Refused as exc:
            raise SystemExit(str(exc))
    elif results_path().exists():
        for line in results_path().read_text().splitlines():
            print(report(json.loads(line)))


if __name__ == "__main__":
    main()
