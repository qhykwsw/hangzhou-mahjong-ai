"""One admitted compute job. All strategy code lives in isolated child processes."""

import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mahjong_ai.execution.candidate_runtime import (
    InlinePolicy,
    JobCancelled,
    check_cancelled,
    is_finalizing,
)
from mahjong_ai.domain.engine import MahjongEngine, derive_seed
from mahjong_ai.domain.settlement import score_ranks
from mahjong_ai.strategies.common import PolicySpec
from mahjong_ai.persistence.artifacts import (
    atomic_json,
    read_json,
    runtime_fingerprint,
)
from mahjong_ai.domain.types import RoundConfig


def _block_means(values_by_block):
    """按 block 求均值；block 是同一副牌、同一批对手的四座轮换组。"""
    return [
        sum(values) / len(values)
        for _, values in sorted(values_by_block.items())
        if values
    ]


def _stats_snapshot(policy):
    """一次性统计基线：复用进程时回执里只能记本场的增量。"""
    return {
        key: value
        for key, value in getattr(policy, "stats", {}).items()
        if isinstance(value, (int, float))
    }


def _stats_delta(policy, snapshot):
    stats = getattr(policy, "stats", {})
    return {
        key: round(value - snapshot.get(key, 0), 6)
        if isinstance(value, float)
        else value - snapshot.get(key, 0)
        for key, value in stats.items()
        if isinstance(value, (int, float))
    }


def _attach_group_readouts(summary, assignments, receipts):
    """Preserve four-seat block means when several blocks share one job."""
    groups = {}
    for assignment, receipt in zip(assignments, receipts):
        block = assignment.get("block")
        if block is None:
            continue
        hands = [tuple(delta) for delta in receipt.get("hand_deltas") or ()]
        hand_ranks = [score_ranks(delta) for delta in hands]
        for seat, candidate in enumerate(receipt["players"]):
            row = groups.setdefault(candidate, {}).setdefault(
                block,
                {
                    "field_sum": 0.0,
                    "field_count": 0,
                    "field_score_sum": 0.0,
                    "hand_sum": 0.0,
                    "hand_count": 0,
                    "score_sum": 0.0,
                },
            )
            row["field_sum"] += 4 - receipt["ranks"][seat]
            row["field_score_sum"] += (receipt.get("scores") or [0] * 4)[seat]
            row["field_count"] += 1
            for delta, ranks in zip(hands, hand_ranks):
                row["hand_sum"] += 4 - ranks[seat]
                row["hand_count"] += 1
                row["score_sum"] += delta[seat]
    for row in summary:
        blocks = groups.get(row["candidate_id"], {})
        row["group_means"] = [
            values["field_sum"] / values["field_count"]
            for _, values in sorted(blocks.items())
            if values["field_count"]
        ]
        # 主口径：每块的场均桌面分（名次效用留作对照）。
        row["field_score_group_means"] = [
            values["field_score_sum"] / values["field_count"]
            for _, values in sorted(blocks.items())
            if values["field_count"]
        ]
        row["hand_group_means"] = [
            values["hand_sum"] / values["hand_count"]
            for _, values in sorted(blocks.items())
            if values["hand_count"]
        ]
        row["score_group_means"] = [
            values["score_sum"] / values["hand_count"]
            for _, values in sorted(blocks.items())
            if values["hand_count"]
        ]


def _build_started_policy(task, key):
    spec = PolicySpec(**task["policies"][key[0]])
    if spec.family == "code" and task.get("resources", {}).get("isolation") == "none":
        # The task owner vouches for its code packages: run them in this process.
        policy = InlinePolicy(spec.package_path, spec.package_hash, spec.action_seconds,
                              spec.version, spec.response_seconds)
    else:
        policy = spec.build()
    # Built-in rule/RL policies run in the host and have no sandbox startup.
    if hasattr(policy, "start"):
        policy.start()
    return policy


_PROGRESS_WRITTEN = [0.0]


def _report_progress(root, completed, total, hand):
    """At most one progress file every two seconds: each write is an fsync."""
    now = time.monotonic()
    if now - _PROGRESS_WRITTEN[0] < 2.0:
        return
    _PROGRESS_WRITTEN[0] = now
    atomic_json(
        root / "progress.json",
        dict(completed_matches=completed, total_matches=total, hand=hand),
    )


def _play_one(task, root, assignment, pool, engine, progress):
    """Play one assignment with this table's policy pool; returns its receipt."""
    path = root / "matches" / f"{assignment['index']:06d}.json"
    if path.exists():
        return read_json(path)
    keys, occurrences = [], {}
    for cid in assignment["players"]:
        slot = occurrences.get(cid, 0)
        occurrences[cid] = slot + 1
        keys.append((cid, slot))
    missing = [key for key in keys if key not in pool]
    if missing:
        # 首次上场要付 sandbox 启动与解释器导入成本，四个进程并行建。
        with ThreadPoolExecutor(max_workers=len(missing)) as executor:
            futures = {
                executor.submit(_build_started_policy, task, key): key
                for key in missing
            }
            built = {}
            try:
                for future in as_completed(futures):
                    built[futures[future]] = future.result()
            except BaseException:
                # A failed parallel startup must not strand the other
                # already-started sandboxes before the task fails.
                for future in futures:
                    if future.done() and not future.cancelled():
                        try:
                            policy = future.result()
                        except BaseException:
                            continue
                        policy.close()
                raise
        pool.update(built)
    players = [pool[key] for key in keys]
    engine.progress = progress
    before = [_stats_snapshot(p) for p in players]
    seed = derive_seed(task["seed"], "wall", assignment["block"])
    seeds = [
        derive_seed(task["seed"], "policy", assignment["block"], seat)
        for seat in range(4)
    ]
    result = engine.run_round(
        players, RoundConfig(**task["config"]), seed, policy_seeds=seeds
    )
    if not result.completed:
        raise ValueError("评估结算不完整")
    receipt = dict(
        players=assignment["players"],
        ranks=result.rank_by_seat,
        scores=result.total_score_by_seat,
        hands=result.rounds_played,
        # 逐局增量分按座位保存：场排名只用了 8 局的总分，逐局读数
        # 让每一局都参与评估，不再把 7/8 的对局信息丢掉。
        hand_deltas=[
            list(hand["delta"]) for hand in result.metadata.get("hands", [])
        ],
        execution=[
            _stats_delta(p, snapshot) for p, snapshot in zip(players, before)
        ],
        diagnostics=[
            dict(candidate_id=cid, error=p.last_error)
            for cid, p in zip(assignment["players"], players)
            if p.last_error
        ],
    )
    atomic_json(path, receipt)
    return receipt


def play(task, root):
    from mahjong_ai.evaluation.schedule import (
        group_uncertainty,
        metric_uncertainty,
        standings,
    )

    # 同一策略在同一任务里反复上场：进程按“策略 + 场上第几次出现”复用，
    # 只在任务结束时收掉。每场重新 build 会按场付出 sandbox 启动与解释器
    # 导入成本（每场约 0.2s×4），对快策略是主要开销。
    #
    # A process plays ``resources.tables`` games at once, one thread each: a
    # model seat's decision is a round trip to the inference service, during
    # which the other tables keep the core busy, and the service sees several
    # requests at a time instead of one.
    assignments = task["assignments"]
    tables = max(1, int(task.get("resources", {}).get("tables") or 1))
    if tables > 1:
        # A table whose model reply has arrived waits for the running table to
        # release the interpreter; the default 5 ms interval would add most of
        # that to every model decision.
        sys.setswitchinterval(0.001)
    receipts = [None] * len(assignments)
    queue = list(range(len(assignments)))
    lock, failures, done = threading.Lock(), [], [0]

    def progress(hand):
        _report_progress(root, done[0], len(assignments), hand)

    def table():
        pool, engine = {}, MahjongEngine(record_events=False)
        try:
            while True:
                with lock:
                    if failures or not queue:
                        return
                    index = queue.pop(0)
                check_cancelled()
                receipts[index] = _play_one(task, root, assignments[index], pool, engine, progress)
                with lock:
                    done[0] += 1
        except BaseException as exc:
            with lock:
                failures.append(exc)
        finally:
            with ThreadPoolExecutor(max_workers=max(1, len(pool))) as executor:
                list(executor.map(lambda policy: policy.close(), pool.values()))

    threads = [threading.Thread(target=table, name=f"table-{i}") for i in range(tables)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]
    roster = list(task["policies"])
    summary = standings(roster, receipts)
    _attach_group_readouts(summary, task["assignments"], receipts)
    result = {"status": "completed", "summary": summary, "matches": len(receipts)}
    if task.get("candidate"):
        # Only expose the focal policy's errors to its research branch. Full
        # tracebacks and other branches' diagnostics stay in host receipts.
        diagnostics = []
        for receipt in receipts:
            for row in receipt.get("diagnostics", []):
                if row["candidate_id"] != task["candidate"]:
                    continue
                error = row["error"]
                entry = {
                    "type": error.get("type"),
                    "message": error.get("message", "")[:2000],
                }
                if entry not in diagnostics:
                    diagnostics.append(entry)
        result["diagnostics"] = diagnostics[:10]
        result["diagnostics_truncated"] = max(0, len(diagnostics) - 10)
        focal_receipts = [
            {
                **receipt,
                "players": [
                    task["candidate"] if seat == assignment["focal"] else "__opponent__"
                    for seat in range(4)
                ],
            }
            for assignment, receipt in zip(task["assignments"], receipts)
        ]
        result["focal"] = standings([task["candidate"]], focal_receipts)[0]
        group_blocks = set()
        groups, field_score_groups, hand_groups, score_groups = {}, {}, {}, {}
        for assignment, receipt in zip(task["assignments"], receipts):
            block, seat = assignment["block"], assignment["focal"]
            group_blocks.add(block)
            groups.setdefault(block, []).append(4 - receipt["ranks"][seat])
            field_score_groups.setdefault(block, []).append(
                (receipt.get("scores") or [0] * 4)[seat]
            )
            for delta in receipt.get("hand_deltas") or ():
                ranks = score_ranks(tuple(delta))
                hand_groups.setdefault(block, []).append(4 - ranks[seat])
                score_groups.setdefault(block, []).append(delta[seat])
        # 逐组读数随回执返回：评估分片合并时按同一口径重算组间标准误。
        blocks = sorted(group_blocks)
        field_means = _block_means(groups)
        field_score_means = _block_means(field_score_groups)
        hand_means = _block_means(hand_groups)
        score_means = _block_means(score_groups)
        result["focal"].update(group_uncertainty(field_means))
        result["focal"].update(
            metric_uncertainty(field_score_means, prefix="field_score_")
        )
        result["focal"].update(metric_uncertainty(hand_means, prefix="hand_"))
        result["focal"].update(metric_uncertainty(score_means, prefix="score_"))
        result["focal"]["group_blocks"] = blocks
        result["focal"]["group_means"] = field_means
        result["focal"]["field_score_group_means"] = field_score_means
        result["focal"]["hand_group_means"] = hand_means
        result["focal"]["score_group_means"] = score_means
    return result


def main():
    root = Path(sys.argv[1])
    task = read_json(root / "task.json")
    if task["runtime"] != runtime_fingerprint():
        raise ValueError("任务必须使用自己的冻结运行时")
    cancel = root / "cancel.flag"
    os.environ["MAHJONG_JOB_CANCEL"] = str(cancel)
    if task["kind"] == "match":
        # Evaluation games want throughput; latency-sensitive work on the same
        # machine (platform play) goes first when the CPU is contended.
        try:
            os.nice(5)
        except OSError:
            pass

    def stop(signum, frame):
        cancel.touch()
        if is_finalizing():
            return
        raise JobCancelled("任务被操作或额度中断")

    signal.signal(signal.SIGTERM, stop)

    def watch_owner():
        while True:
            time.sleep(0.2)
            if os.getppid() != task["owner_pid"] or time.time() >= task["deadline"]:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch_owner, daemon=True).start()
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        check_cancelled()
        value = play(task, root)
    except Exception as exc:
        # Teardown after an interruption can raise something else (a sandbox
        # already gone, EPERM); the cancel flag says the job was interrupted.
        if not isinstance(exc, JobCancelled) and not cancel.exists():
            value = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:4000]}
        elif time.time() >= task["deadline"]:
            value = {"status": "failed", "error": "iteration execution limit reached"}
        else:
            value = {"status": "paused", "reason": "operator interruption"}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    value["usage"] = dict(
        wall_seconds=time.monotonic() - started,
        host_cpu_seconds=time.process_time(),
        child_cpu_seconds=after.ru_utime
        + after.ru_stime
        - before.ru_utime
        - before.ru_stime,
    )
    atomic_json(root / "result.json", value)


if __name__ == "__main__":
    main()
