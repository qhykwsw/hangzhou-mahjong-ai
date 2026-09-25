"""The iteration agent's loop over a line (ADR-0016, handbooks/common.md).

Any session of any model drives a line with these commands; nothing depends
on the conversation that started an attempt:

  python -m harness.line wait rule|rl [--minutes 9]   # results, notices, and the next step
  python -m harness.line start rule|rl "<change>" --why "<hypothesis>" --design-start <epoch> [--host <free host>]
  python -m harness.line status rule|rl               # every attempt, the hosts, the lease
  python -m harness.line stop rule|rl                 # leave the loop; attempts run on

Only on the user's explicit word:
  python -m harness.line takeover rule|rl
  python -m harness.line hosts [on|off <host> [--now]]
  python -m harness.line cancel <attempt>
  python -m harness.line promote <attempt>

Each attempt runs in a detached driver (``drive``) whose whole state is
.local/running/<id>.json. A driver that dies (its session closed, the Mac
restarted) is started again by the next wait and resumes where it stopped,
while the attempt's remote training or evaluation keeps running on its host.
A host that restarts and loses an attempt's training puts the attempt back
in the queue, ahead of new designs, for any free host to redo. One session
at a time holds a line's lease; every wait renews it.
"""
import argparse
import fcntl
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import evaluate as E
from . import rl
from guided import recipe_jobs, remote  # noqa: E402  (rl puts guided on the path)

LINE_HOSTS = {"rule": ("local",), "rl": rl.HOSTS}
RUNNING = E.LOCAL / "running"
LEASE_SECONDS = 20 * 60  # a lease not renewed this long is free for another session
PROBE_SECONDS = 120  # host answers are reused this long
POLL_SECONDS = 15
QUEUE_SECONDS = 30


# -- attempt state ---------------------------------------------------------------
def load(attempt_id):
    path = RUNNING / f"{attempt_id}.json"
    return E.read_json(path) if path.exists() else None


def save(state):
    E.atomic_json(RUNNING / f"{state['id']}.json", state)


def queued(line):
    return sorted((s for s in E.running(line) if s.get("stage") == "queued"), key=lambda s: s["started_at"])


def extra(state, training=None):
    """What an attempt's record carries besides its evaluation."""
    return {"why": state.get("why"), "base": state.get("base"), "started_at": state["started_at"],
            "training": training or state.get("training")}


def record_failure(state, status, error, training=None):
    """Record an attempt that ends without an evaluation (crash, invalid, cancelled)."""
    record = dict(id=state["id"], line=state["line"], pool=E.load_pool()["id"], status=status,
                  description=state["description"], host=state.get("host"), finished_at=time.time(),
                  games=0, score=None, se=None, best_before=None, diff=None, confirm=None, stats={},
                  spec=state.get("spec"), dev_blocks={}, design_seconds=state.get("design_seconds"),
                  error=error)
    E.append_record(record | extra(state, training))


# -- hosts -----------------------------------------------------------------------
def settings():
    path = E.LOCAL / "hosts.json"
    return E.read_json(path) if path.exists() else {}


def enabled(host):
    """The user's switch for a host (hosts on|off); every host starts switched on."""
    return host == "local" or settings().get(host, {}).get("enabled", True)


def online(hosts, probe=True):
    """Which hosts answer, probed in parallel; answers are reused for PROBE_SECONDS."""
    path = E.LOCAL / "probe.json"
    cache = E.read_json(path) if path.exists() else {}
    stale = [h for h in hosts if h != "local" and time.time() - cache.get(h, {}).get("at", 0) > PROBE_SECONDS]
    if probe and stale:
        with ThreadPoolExecutor(len(stale)) as pool:
            answers = list(pool.map(recipe_jobs.reachable, stale))
        cache = (E.read_json(path) if path.exists() else {}) | {
            h: {"online": ok, "at": time.time()} for h, ok in zip(stale, answers)}
        E.atomic_json(path, cache)
    return {h: h == "local" or cache.get(h, {}).get("online", False) for h in hosts}


def ready(host):
    return host == "local" or rl.setup_path(host).exists()


def free_hosts(line, probe=True):
    """Switched-on, set-up hosts of the line with no attempt on them that answer now."""
    busy = {s.get("host") for s in E.running(line)}
    hosts = [h for h in LINE_HOSTS[line] if enabled(h) and ready(h) and h not in busy]
    up = online(hosts, probe)
    return [h for h in hosts if up[h]]


def design_hosts(line, probe=True):
    """(the hosts a new attempt may start on now, None), else ([], why not).

    Attempts queued for a redo take the first free hosts.
    """
    champion = E.best(E.load_pool()["id"], line)
    if champion is None and E.running(line):
        return [], "第一次尝试（v0）还没有结果，它成为最好版本后才能设计新尝试"
    free, waiting = free_hosts(line, probe), queued(line)
    if len(free) > len(waiting):
        return free[len(waiting):], None
    if line == "rule":
        return [], "规则线的尝试还在评估"
    if waiting:
        return [], f"{len(waiting)} 个尝试在排队等主机重做，它们优先"
    return [], "没有空闲的在线主机"


def host_states(line):
    busy = {s["host"]: s["id"] for s in E.running(line) if s.get("host")}
    hosts = LINE_HOSTS[line]
    up = online([h for h in hosts if enabled(h)])
    states = {}
    for h in hosts:
        if not enabled(h):
            states[h] = "已关闭"
        elif not ready(h):
            states[h] = "未起步"
        else:
            states[h] = "在线" if up[h] else "离线"
    return states, busy


def host_notices(line):
    """Host changes since the last notice (online, offline, switched off), for the user."""
    if line == "rule":
        return []
    states, busy = host_states(line)
    path = E.LOCAL / f"notices-{line}.json"
    before = E.read_json(path) if path.exists() else {}
    notices = []
    for h, now in states.items():
        was = before.get(h, "在线")
        if now != was:
            waiting = f"，在途尝试 {busy[h]} 等它回来" if now == "离线" and h in busy else ""
            notices.append(f"{h}：{was} → {now}{waiting}")
    if before != states:
        E.atomic_json(path, states)
    if notices and not any(s == "在线" for s in states.values()):
        notices.append("所有主机都不可用，RL 线暂时无法开新尝试")
    return notices


# -- lease -----------------------------------------------------------------------
def ps(pid):
    out = subprocess.run(["ps", "-o", "ppid=,lstart=,comm=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    ppid, rest = out.split(None, 1)
    return int(ppid), rest[:24], rest[24:].strip()


def agent_identity():
    """The agent process this command runs under (Claude Code, Codex), or the top of its chain."""
    pid, last = os.getppid(), None
    for _ in range(16):
        row = ps(pid)
        if row is None:
            break
        ppid, started, command = row
        last = {"pid": pid, "start": started, "name": Path(command).name}
        if re.search(r"claude|codex", last["name"], re.I) or ppid <= 1:
            break
        pid = ppid
    return last or {"pid": os.getppid(), "start": "", "name": "unknown"}


def alive(owner):
    row = ps(owner["pid"])
    return row is not None and row[1] == owner["start"]


def claim(line, force=False):
    """Hold or renew the line's lease for this session; refuses while another session holds it."""
    me = agent_identity()
    path = E.LOCAL / f"lease-{line}.json"
    with E.attempts_lock():
        lease = E.read_json(path) if path.exists() else None
        if lease and lease["owner"] != me and not force:
            idle = time.time() - lease["renewed"]
            if idle < LEASE_SECONDS and alive(lease["owner"]):
                o = lease["owner"]
                raise E.Refused(f"{E.LINE_NAMES[line]}正由另一个会话推进（{o['name']} pid {o['pid']}，"
                                f"{idle / 60:.0f} 分钟前续约）。用户明确说「接管」时才运行 takeover。")
        E.atomic_json(path, {"owner": me, "renewed": time.time()})


def release(line):
    path = E.LOCAL / f"lease-{line}.json"
    with E.attempts_lock():
        if path.exists() and E.read_json(path)["owner"] == agent_identity():
            path.unlink()


# -- the line's directory ----------------------------------------------------------
def git(*args):
    return subprocess.run(["git", *args], cwd=E.REPO, capture_output=True, text=True)


def restore(line):
    """Put the line's best version in its directory and commit it if it changed there.

    Runs before every design, so an attempt is always one change from the best
    version, whatever the directory held (another attempt's change, a crashed
    session's half-done edit). The base is remembered for start().
    """
    champion = E.best(E.load_pool()["id"], line)
    target = E.SOURCES[line]
    rel = target.relative_to(E.REPO).as_posix()
    if champion:
        shutil.rmtree(target)
        shutil.copytree(E.frozen_source(champion), target, ignore=shutil.ignore_patterns("__pycache__"))
        commit(rel, f"{E.LINE_NAMES[line]}：{champion['id']} {champion['description']}")
    else:
        git("checkout", "HEAD", "--", rel)
        git("clean", "-fdq", "--", rel)
    E.atomic_json(E.LOCAL / f"workdir-{line}.json", {"base": champion["id"] if champion else None, "at": time.time()})
    return champion


def commit(rel, message):
    if E.TEST:
        return  # a pipeline check never commits
    git("add", "-A", "--", rel)
    if git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
        return
    for _ in range(10):
        result = git("commit", "-q", "-m", message, "--", rel)
        if result.returncode == 0:
            return
        time.sleep(3)  # another commit holds the index
    E.log(f"提交 {rel} 失败：{result.stderr.strip()[-300:]}")


# -- drivers ---------------------------------------------------------------------
def spawn(attempt_id):
    (E.LOCAL / "logs").mkdir(parents=True, exist_ok=True)
    with (E.LOCAL / "logs" / f"{attempt_id}.log").open("ab") as out:
        subprocess.Popen([sys.executable, "-m", "harness.line", "drive", attempt_id], cwd=E.HERE,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)


def driver_alive(attempt_id):
    RUNNING.mkdir(parents=True, exist_ok=True)
    with (RUNNING / f"{attempt_id}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def ensure_drivers(line):
    for state in E.running(line):
        if not driver_alive(state["id"]):
            spawn(state["id"])


def drive(attempt_id):
    """Carry one attempt to its record, from whatever stage it is in."""
    RUNNING.mkdir(parents=True, exist_ok=True)
    handle = (RUNNING / f"{attempt_id}.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # another driver has it
    state = load(attempt_id)
    if state is None:
        return
    state["driver_pid"] = os.getpid()
    save(state)
    E.log(f"{attempt_id}: driver {os.getpid()} at stage {state['stage']}")
    try:
        E.check_seal()
        advance(state)
    except E.Refused as exc:
        E.log(f"{attempt_id}: {exc}")  # resumes once the maintainer reseals
    except Exception:
        trace = traceback.format_exc()
        E.log(trace)
        state = load(attempt_id)
        if state is not None:
            record_failure(state, "crash", trace[-2000:])


def advance(state):
    while state is not None:
        if state["stage"] == "queued":
            state = claim_host(state)
        elif state["stage"] == "training":
            state = train(state)
        else:
            E.evaluate(state["line"], state["spec"], state["description"], state["id"],
                       state.get("design_seconds"), host=state["host"],
                       confirm_seed=state["confirm_seed"], extra=extra(state))
            return


def train(state):
    host = state["host"]
    try:
        outcome, spec, training = rl.train(state)
    except recipe_jobs.HostFault as exc:
        E.log(f"{state['id']}: {exc}; back in the queue")
        return requeue(state, "training")
    except Exception:
        if recipe_jobs.reachable(host):
            raise
        E.log(f"{state['id']}: {host} unreachable; waiting for it")
        while not recipe_jobs.reachable(host):
            time.sleep(60)
        return load(state["id"])  # the launch file lets the training be picked up again
    if outcome != "ok":
        record_failure(state, outcome, training.pop("error"), training)
        return None
    state = load(state["id"]) | {"stage": "evaluating", "spec": spec, "training": training}
    save(state)
    return state


def requeue(state, stage):
    """The attempt's host lost its work or was taken away: any free host redoes the stage."""
    state = load(state["id"]) | {"stage": "queued", "next": stage, "host": None,
                                 "requeued": state.get("requeued", 0) + 1}
    if stage == "training":
        (rl.RL / "attempts" / state["id"] / "launch-train.json").unlink(missing_ok=True)
    else:
        drop_evaluation(state["id"])
    save(state)
    return state


def evaluation_dirs(attempt_id):
    root = E.LOCAL / "jobs" / E.load_pool()["id"]
    return [root / attempt_id, *root.glob(f"*/confirm-for-{attempt_id}-*")]


def drop_evaluation(attempt_id):
    """Evaluation jobs split by the host's workers: another host starts them over."""
    for path in evaluation_dirs(attempt_id):
        shutil.rmtree(path, ignore_errors=True)


def claim_host(state):
    """Wait in the queue (oldest first) for a free host, then take it."""
    while True:
        online(LINE_HOSTS[state["line"]])
        with E.attempts_lock():
            current = load(state["id"])
            if current is None or current["stage"] != "queued":
                return current
            free, waiting = free_hosts(state["line"], probe=False), queued(state["line"])
            if free and waiting[0]["id"] == state["id"]:
                current |= {"stage": current["next"], "host": free[0]}
                save(current)
                E.log(f"{state['id']}: redoing {current['stage']} on {free[0]}")
                return current
        time.sleep(QUEUE_SECONDS)


def interrupt(state):
    """Stop an attempt's driver and its jobs on its host (to redo it elsewhere, or cancel it)."""
    if state.get("driver_pid") and driver_alive(state["id"]):
        try:
            os.kill(state["driver_pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(50):
            if not driver_alive(state["id"]):
                break
            time.sleep(0.2)
    host = state.get("host")
    if host in (None, "local"):
        return  # local jobs pause themselves when their driver goes
    pids = []
    launch = rl.RL / "attempts" / state["id"] / "launch-train.json"
    if launch.exists():
        pids.append(E.read_json(launch)["pid"])
    for root in evaluation_dirs(state["id"]):
        pids += [E.read_json(m)["pid"] for m in root.glob("*/remote.json")] + (
            [E.read_json(root / "remote.json")["pid"]] if (root / "remote.json").exists() else [])
    if pids:
        kill_later(host, state["id"], pids)
        retry_kills()


def kill_later(host, attempt_id, pids):
    path = E.LOCAL / "pending-kills.json"
    with E.attempts_lock():
        rows = E.read_json(path) if path.exists() else []
        E.atomic_json(path, rows + [{"host": host, "attempt": attempt_id, "pids": pids}])


def retry_kills():
    """Kill jobs left on hosts that were offline when their attempt was stopped.

    A process is killed only while its command line still names the attempt,
    so a pid reused on the host after a restart is left alone.
    """
    path = E.LOCAL / "pending-kills.json"
    if not path.exists():
        return
    left = []
    for row in E.read_json(path):
        script = f"""import os,signal
for pid in {row['pids']!r}:
    try:
        if {row['attempt']!r} in open(f'/proc/{{pid}}/cmdline','rb').read().decode(errors='replace'):
            os.killpg(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
print('ok')
"""
        try:
            remote.execute(row["host"], script, timeout=60)
        except Exception:
            left.append(row)
    with E.attempts_lock():
        E.atomic_json(path, left)


# -- the agent's commands ------------------------------------------------------------
def preflight(line):
    E.check_seal()
    if line == "rl" and not any(ready(h) for h in rl.HOSTS):
        raise E.Refused("RL 线还没有任何主机完成起步（harness.rl setup），这是维护者的事。")
    if line == "rl" and E.best(E.load_pool()["id"], "rl") is None:
        raise E.Refused("RL 线在当前对手池下还没有起点成绩（harness.rl baseline），这是维护者的事。")


def summary(line):
    champion = E.best(E.load_pool()["id"], line)
    out = [f"最好版本：{champion['id']} {champion['score']:+.2f}｜{champion['description']}" if champion
           else "最好版本：还没有"]
    for s in E.running(line):
        minutes = (time.time() - s["started_at"]) / 60
        where = f" @{s['host']}" if s.get("host") else ""
        out.append(f"在途：{s['id']} {E.STAGE_NAMES.get(s.get('stage'), '进行中')}{where} {minutes:.0f} 分｜{s['description']}")
    if line == "rl":
        states, busy = host_states(line)
        out.append("主机：" + "；".join(f"{h} {v}" + (f"（{busy[h]}）" if h in busy else "") for h, v in states.items()))
    return out


def notes_path(line):
    return E.LOCAL / "notes" / f"{line}.md"


def emit(records, notices, step, line):
    blocks = []
    if records:
        blocks.append("【汇报】（原样发给用户）\n" + "\n\n".join(E.report(r) for r in records))
    if notices:
        blocks.append("【通知】（原样发给用户）\n" + "\n".join(notices))
    blocks.append(f"【下一步】{step}")
    try:
        blocks.append("【状态】\n" + "\n".join(summary(line)))
    except Exception:
        pass
    print("\n\n".join(blocks))


def wait(line, minutes):
    """Block until there is something to report or to design, at most ``minutes``."""
    deadline = time.time() + minutes * 60
    while True:
        try:
            claim(line)
            preflight(line)
        except E.Unsealed as exc:
            # The maintainer changes the conditions and reseals right after;
            # only a mismatch that outlasts this wait is theirs to settle.
            if time.time() < deadline:
                time.sleep(POLL_SECONDS)
                continue
            return emit([], [], f"停下：{exc}", line)
        except (E.Refused, FileNotFoundError) as exc:
            return emit([], [], f"停下：{exc}", line)
        ensure_drivers(line)
        retry_kills()
        pool_id = E.load_pool()["id"]
        done = E.reported()
        records = [r for r in E.attempts(pool_id, line) if r["id"] not in done]
        notices = host_notices(line)
        flagged = [r["id"] for r in records if (r.get("reference") or {}).get("above")]
        free, why_not = design_hosts(line)
        if flagged:
            E.mark_reported(records)
            return emit(records, notices, f"停下：{'、'.join(flagged)} 高于自由对战的参照线。问用户是否换上；"
                                          "用户同意就运行 promote <编号>，然后继续循环。", line)
        if free or records or notices or time.time() >= deadline:
            E.mark_reported(records)
            if free:
                champion = restore(line)
                base = (f"本线目录已恢复为最好版本 {champion['id']}" if champion
                        else "还没有最好版本，第一次尝试用本线目录原样（v0）")
                where, command = "", "start"
                if line == "rl":
                    where = (f"空闲主机 {free[0]}，按它的资源设计" if len(free) == 1
                             else f"空闲主机 {'、'.join(free)}，选一台按它的资源设计") + "（见 status 的主机资源）；"
                    command = "start 并用 --host 指定这台主机"
                step = f"设计：{where}{base}。先读 status 与迭代笔记 {notes_path(line)}，只改一处，然后运行 {command}。"
            else:
                step = f"等待：{why_not}。再次运行 wait。"
            return emit(records, notices, step, line)
        time.sleep(POLL_SECONDS)


def start(line, change, why, design_start, host=None):
    claim(line)
    preflight(line)
    champion = E.best(E.load_pool()["id"], line)
    source = E.SOURCES[line]
    rel = source.relative_to(E.REPO).as_posix()
    workdir = E.LOCAL / f"workdir-{line}.json"
    base = E.read_json(workdir).get("base") if workdir.exists() else None
    if champion:
        if base != champion["id"]:
            raise E.Refused(f"最好版本现在是 {champion['id']}，本线目录不是从它恢复来的。"
                            "运行 wait 恢复后，把这处改动重新做在它上面。")
        # The start was never trained with its recipe, so training that recipe unchanged is an attempt too.
        if champion["id"] != rl.start_id(E.load_pool()) and E.tree_hash(source) == E.tree_hash(E.frozen_source(champion)):
            raise E.Refused("本线目录与最好版本完全相同，没有可评估的改动。")
    elif git("status", "--porcelain", "--", rel).stdout.strip():
        raise E.Refused("还没有最好版本：第一次尝试必须是本线目录原样（v0）。运行 wait 恢复后再开跑。")
    problems = rl.check_recipe(source) if line == "rl" else E.check_source(source)
    if problems:
        raise E.Refused("不符合迭代线约定（进程、线程与资源归维护者），没有开跑：\n" + "\n".join(problems))
    online(LINE_HOSTS[line])
    with E.attempts_lock():
        free, why_not = design_hosts(line, probe=False)
        if not free:
            raise E.Refused(f"现在不能开新尝试：{why_not}")
        host = host or free[0]
        if host not in free:
            raise E.Refused(f"{host} 现在不能开新尝试（可用：{'、'.join(free)}）。运行 wait，按可用的主机调整设计。")
        attempt_id = E.next_attempt_id(line)
        now = time.time()
        state = dict(id=attempt_id, line=line, host=host, description=change, why=why,
                     base=champion["id"] if champion else None, started_at=now,
                     design_seconds=now - design_start, stage="training" if line == "rl" else "evaluating",
                     seed=random.SystemRandom().getrandbits(31), confirm_seed=random.SystemRandom().getrandbits(62))
        save(state)
    if line == "rule":
        state["spec"] = E.freeze_rule(attempt_id)
    else:
        rl.snapshot(attempt_id)
    save(state)
    spawn(attempt_id)
    number = int(attempt_id.rsplit("-", 1)[-1])
    print(f"尝试 {number} · {E.LINE_NAMES[line]}：{change}" + (f"（{host}）" if line == "rl" else ""))


def status(line):
    print(E.history(line))
    print("\n".join(row for row in summary(line) if row.startswith("主机")))
    if line == "rl":
        print(rl.host_resources())
    path = E.LOCAL / f"lease-{line}.json"
    if path.exists():
        lease = E.read_json(path)
        o = lease["owner"]
        print(f"推进权：{o['name']} pid {o['pid']}，{(time.time() - lease['renewed']) / 60:.0f} 分钟前续约")
    print(f"迭代笔记：{notes_path(line)}" + ("" if notes_path(line).exists() else "（还没有，第一次拿到结果后新建）"))


def hosts(action=None, host=None, now=False):
    if action:
        if host not in rl.HOSTS:
            raise E.Refused(f"主机只能是 {'、'.join(rl.HOSTS)}")
        with E.attempts_lock():
            values = settings()
            values.setdefault(host, {})["enabled"] = action == "on"
            E.atomic_json(E.LOCAL / "hosts.json", values)
        if action == "off" and now:
            for state in E.running("rl"):
                if state.get("host") == host:
                    interrupt(state)
                    stage = "training" if state["stage"] == "training" else "evaluating"
                    requeue(state, stage)
                    print(f"{state['id']} 已从 {host} 撤下，排队到其他空闲主机重做{'训练' if stage == 'training' else '评估'}")
    states, busy = host_states("rl")
    for h, v in states.items():
        print(f"{h}：{v}" + (f"，在途 {busy[h]}" if h in busy else ""))


def cancel(attempt_id):
    state = load(attempt_id)
    if state is None:
        raise E.Refused(f"{attempt_id} 不是在途尝试")
    interrupt(state)
    record_failure(load(attempt_id), "cancelled", "用户取消")
    print(f"{attempt_id} 已取消")


def promote(attempt_id):
    """Put a confirmed best version into platform free play (the resident script picks up the pin)."""
    if E.TEST:
        raise E.Refused("流程检查不改动自由对战")
    pool = E.load_pool()
    rows = [r for r in E.attempts(pool["id"]) if r["id"] == attempt_id and r["status"] == "keep"]
    if not rows:
        raise E.Refused("只能换上当前对手池下确认成功的版本")
    record = rows[-1]
    target = E.REPO / ".local/platform" / f"entrant-{attempt_id}"
    if not (target / "policy.json").exists():
        export(record, pool, target)
    pin = E.REPO / ".local/competition/entrant.pin"
    before = E.platform_entrant()
    pin.write_text(attempt_id + "\n")
    playing = subprocess.run(["pgrep", "-f", "free_play_strongest"], capture_output=True).returncode == 0
    print(f"自由对战上场版本：{before} → {attempt_id}。"
          + ("常驻脚本在下一房开始时换上。" if playing else "常驻脚本没有在运行，启动后才会上场。"))


def export(record, pool, target):
    tmp = target.with_name(target.name + ".incoming")
    shutil.rmtree(tmp, ignore_errors=True)
    spec = record["spec"]
    if spec["family"] == "model":
        weights = spec["parameters"]["weights"]
        shutil.copytree(E.MODELS / weights, tmp / "models" / weights)
        policy = {"family": "model", "version": record["id"], "parameters": {"weights": weights}}
    else:
        shutil.copytree(spec["package_path"], tmp / "candidate")
        shutil.copytree(pool["runtime_path"], tmp / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        policy = E.with_timeouts(spec) | {"package_path": "candidate", "python_path": None, "runtime_path": "runtime"}
    E.atomic_json(tmp / "policy.json", policy)
    E.atomic_json(tmp / "manifest.json", {"attempt": record["id"], "line": record["line"], "pool": pool["id"],
                                          "runtime": Path(pool["runtime_path"]).name, "family": spec["family"],
                                          "scope": "local iteration-line result; the platform keeps its own records"})
    tmp.rename(target)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lines = ("rule", "rl")
    w = sub.add_parser("wait")
    w.add_argument("line", choices=lines)
    w.add_argument("--minutes", type=float, default=9)
    s = sub.add_parser("start")
    s.add_argument("line", choices=lines)
    s.add_argument("change", help="one line: what this attempt changes")
    s.add_argument("--why", required=True, help="one or two sentences: the hypothesis behind it")
    s.add_argument("--design-start", type=float, required=True, help="epoch when designing this attempt began")
    s.add_argument("--host", help="RL: the free host this attempt was designed for (wait lists them)")
    for name in ("status", "stop", "takeover"):
        sub.add_parser(name).add_argument("line", choices=lines)
    h = sub.add_parser("hosts")
    h.add_argument("action", nargs="?", choices=("on", "off"))
    h.add_argument("host", nargs="?")
    h.add_argument("--now", action="store_true", help="with off: take the attempt away from the host at once")
    for name in ("cancel", "promote", "drive"):
        sub.add_parser(name).add_argument("attempt")
    args = parser.parse_args()
    try:
        if args.command == "wait":
            wait(args.line, args.minutes)
        elif args.command == "start":
            start(args.line, args.change, args.why, args.design_start, args.host)
        elif args.command == "status":
            status(args.line)
        elif args.command == "stop":
            release(args.line)
            print(f"{E.LINE_NAMES[args.line]}：已退出循环；在途尝试继续跑，结果在下次 wait 时汇报")
        elif args.command == "takeover":
            claim(args.line, force=True)
            print(f"{E.LINE_NAMES[args.line]}：本会话已接管；原会话下次调用时会停下")
        elif args.command == "hosts":
            hosts(args.action, args.host, args.now)
        elif args.command == "cancel":
            cancel(args.attempt)
        elif args.command == "promote":
            promote(args.attempt)
        else:
            drive(args.attempt)
    except E.Refused as exc:
        print(f"拒绝：{exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
