"""Finite Mac-to-WSL teacher publication; completed research is the only source."""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

from . import artifacts, remote


class DeadlineReached(BaseException):
    """Not swallowed by per-host retry handling."""


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _until(deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise DeadlineReached()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(*_):
        raise DeadlineReached()

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, old_timer[0] - (time.monotonic() - started)),
                old_timer[1],
            )


# This program is run by the target Conda Python. Its validation subprocess must
# bind the immutable SDK before importing the actual engine or candidate code.
_VALIDATE = r"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'projects/personal/selfplay'))
from guided.artifacts import read_bundle,bind_runtime,build_policy
bundle=read_bundle(sys.argv[1]);bind_runtime(bundle)
from mahjong_ai.domain.engine import HandState,MahjongEngine
from mahjong_ai.domain.types import RoundConfig
from mahjong_ai import _fast
policies=[build_policy(bundle,e['id']) for e in bundle['entries']]
if len({id(p) for p in policies}) != 4: raise ValueError('independent policies required')
for i,p in enumerate(policies):
 p.reset(710+i)
 observation=HandState(RoundConfig(rounds=1),710+i).observation()
 if p.act(observation) not in observation.legal_actions: raise ValueError('illegal preflight action')
class Checked:
 def __init__(self,p): self.p=p
 def reset(self,seed): self.p.reset(seed)
 def act(self,obs):
  try: action=self.p.act(obs)
  except Exception as error: raise ValueError('candidate action failed') from error
  if action not in obs.legal_actions: raise ValueError('illegal game action')
  return action
 def finish_round(self,result,seat): self.p.finish_round(result,seat)
result=MahjongEngine(record_events=False).run_round([Checked(p) for p in policies],RoundConfig(rounds=1,allow_tiebreak=False),719)
if not result.completed or result.rounds_played != 1: raise ValueError('incomplete validation game')
if 'torch' in sys.modules: raise ValueError('rule validation imported torch')
print(json.dumps({'id':bundle['id'],'runtime_hash':bundle['runtime_hash'],'hands':result.rounds_played,'policies':len(policies)}))
"""


_PREPARE = r"""
import hashlib,json,os,shutil,signal,subprocess,sys,sysconfig,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'projects/personal/selfplay'))
from guided.artifacts import read_bundle
manifest=Path(sys.argv[1]);deadline=float(sys.argv[2]);validate=sys.argv[3]
if time.time() >= deadline: raise TimeoutError('sync deadline reached')
def expired(*_): raise TimeoutError('sync deadline reached')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
bundle=read_bundle(manifest);runtime=Path(bundle['runtime'])
target=runtime/'mahjong_ai'/('_fast'+sysconfig.get_config_var('EXT_SUFFIX'))
relative=target.relative_to(manifest.parent).as_posix()
if relative not in bundle['files']:
 from setuptools import Extension,Distribution
 with tempfile.TemporaryDirectory(prefix='guided-native-') as folder:
  build=Path(folder)
  dist=Distribution({'ext_modules':[Extension('mahjong_ai._fast',sources=[str(runtime/'mahjong_ai/_fast.c')],extra_compile_args=['-O3'])]})
  cmd=dist.get_command_obj('build_ext');cmd.build_lib=str(build/'lib');cmd.build_temp=str(build/'temp')
  dist.run_command('build_ext')
  output=Path(cmd.get_ext_fullpath('mahjong_ai._fast'))
  temporary=target.with_name('.'+target.name+'.building')
  try:
   shutil.copyfile(output,temporary);os.replace(temporary,target)
  finally:
   if temporary.exists(): temporary.unlink()
read_bundle(manifest)
checked=subprocess.run([sys.executable,'-c',validate,str(manifest)],capture_output=True,text=True,timeout=max(0.01,min(90,deadline-time.time())))
if checked.returncode: raise RuntimeError('frozen policy validation failed')
receipt=json.loads(checked.stdout.strip().splitlines()[-1])
if receipt['id'] != bundle['id'] or receipt['runtime_hash'] != bundle['runtime_hash']: raise ValueError('validation identity mismatch')
receipt.update(validated_at=time.time(),native={relative:hashlib.sha256(target.read_bytes()).hexdigest()})
receipts=manifest.parent.parent/'.receipts';receipts.mkdir(exist_ok=True)
fd,temp=tempfile.mkstemp(prefix='.'+bundle['id'],dir=receipts)
try:
 with os.fdopen(fd,'w') as stream:
  json.dump(receipt,stream);stream.flush();os.fsync(stream.fileno())
 os.replace(temp,receipts/(manifest.parent.name+'.json'))
finally:
 if os.path.exists(temp): os.unlink(temp)
print(json.dumps(receipt))
"""


_PUBLISH = r"""
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'projects/personal/selfplay'))
from guided.artifacts import read_bundle
manifest=Path(sys.argv[1]);expected=sys.argv[2];runtime_hash=sys.argv[3];deadline=float(sys.argv[4])
bundle=read_bundle(manifest)
root=manifest.parent.parent;receipt_path=root/'.receipts'/(manifest.parent.name+'.json')
receipt=json.loads(receipt_path.read_text())
if bundle['id'] != expected or bundle['runtime_hash'] != runtime_hash or receipt['id'] != expected or receipt['runtime_hash'] != runtime_hash: raise ValueError('publication identity mismatch')
for relative,digest in receipt['native'].items():
 path=manifest.parent/relative
 if not path.resolve().is_relative_to(manifest.parent.resolve()) or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('derived native changed since validation')
if receipt['hands'] < 1 or receipt['policies'] != 4: raise ValueError('incomplete validation receipt')
if time.time() >= deadline: raise TimeoutError('sync deadline reached')
destination=root/expected
if destination.exists():
 existing=read_bundle(destination/'manifest.json')
 prior_path=root/'.receipts'/(expected+'.json')
 if not prior_path.exists():
  # Initial/manual transfers may predate synchronization receipts. Preserve
  # every original byte; only add a missing derived extension, then validate
  # the actual existing destination in its own fresh interpreter.
  for relative,digest in receipt['native'].items():
   existing_native=destination/relative
   if not existing_native.resolve().is_relative_to(destination.resolve()) or existing_native.is_symlink(): raise ValueError('invalid existing native destination')
   if not existing_native.exists():
    if relative in existing['files']: raise ValueError('cannot replace missing original native source')
    fd,native_temp=tempfile.mkstemp(prefix='.native-install-',dir=root);os.close(fd)
    try:
     shutil.copyfile(manifest.parent/relative,native_temp);os.replace(native_temp,existing_native)
    finally:
     if os.path.exists(native_temp): os.unlink(native_temp)
  checked=subprocess.run([sys.executable,'-c',sys.argv[5],str(destination/'manifest.json')],capture_output=True,text=True,timeout=max(0.01,min(90,deadline-time.time())))
  if checked.returncode: raise ValueError('existing destination validation failed')
  prior=json.loads(checked.stdout.strip().splitlines()[-1])
  prior.update(validated_at=time.time(),native={relative:hashlib.sha256((destination/relative).read_bytes()).hexdigest() for relative in receipt['native']})
  if prior.get('id') != expected or prior.get('runtime_hash') != runtime_hash or prior.get('hands',0) < 1 or prior.get('policies') != 4: raise ValueError('existing validation identity mismatch')
  fd,prior_temp=tempfile.mkstemp(prefix='.receipt-',dir=root/'.receipts')
  try:
   with os.fdopen(fd,'w') as stream:
    json.dump(prior,stream);stream.flush();os.fsync(stream.fileno())
   os.replace(prior_temp,prior_path)
  finally:
   if os.path.exists(prior_temp): os.unlink(prior_temp)
 prior=json.loads(prior_path.read_text())
 if existing['id'] != expected or prior['id'] != expected or prior['runtime_hash'] != runtime_hash: raise ValueError('existing published identity mismatch')
 for relative,digest in prior['native'].items():
  existing_native=destination/relative
  if not existing_native.resolve().is_relative_to(destination.resolve()) or existing_native.is_symlink() or hashlib.sha256(existing_native.read_bytes()).hexdigest() != digest: raise ValueError('existing native receipt mismatch')
 shutil.rmtree(manifest.parent);receipt_path.unlink()
else:
 os.rename(manifest.parent,destination)
 os.replace(receipt_path,root/'.receipts'/(expected+'.json'))
fd,temp=tempfile.mkstemp(prefix='.current-',dir=root)
try:
 with os.fdopen(fd,'w') as stream:
  json.dump({'id':expected,'manifest':expected+'/manifest.json'},stream);stream.flush();os.fsync(stream.fileno())
 if time.time() >= deadline: raise TimeoutError('sync deadline reached')
 os.replace(temp,root/'current.json')
finally:
 if os.path.exists(temp): os.unlink(temp)
print(json.dumps({'id':expected,'runtime_hash':runtime_hash,'status':'published'}))
"""


_DISCARD = r"""
import shutil,sys
from pathlib import Path
staging=Path(sys.argv[1]);root=Path(sys.argv[2])
if not staging.name.startswith('.incoming-') or staging.parent.resolve() != root.resolve() or staging.is_symlink(): raise ValueError('invalid staging cleanup')
if staging.exists(): shutil.rmtree(staging)
receipt=root/'.receipts'/(staging.name+'.json')
if receipt.exists(): receipt.unlink()
"""


def _record(store, report):
    row = {"time": time.time(), **report}
    with (store / "sync-log.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def sync_once(
    source,
    local_store,
    remote_store,
    *,
    expected_runtime_hash=None,
    hosts=("4090", "4070"),
    deadline=None,
    code_path=remote.ROOT,
):
    """Retry unsynchronized hosts; never publish before native and gameplay checks.

    Runtime incompatibility is rejected before transfer. Receipts contain only
    immutable identities and validation results, never remote command output.
    """
    if any(host not in remote.HOSTS for host in hosts):
        raise ValueError("unknown teacher destination")
    if not Path(remote_store).is_absolute() or ".." in Path(remote_store).parts:
        raise ValueError("remote store must be an absolute path without traversal")
    if not Path(code_path).is_absolute() or ".." in Path(code_path).parts:
        raise ValueError("remote code path must be an absolute path without traversal")
    if deadline is not None and time.time() >= deadline:
        return []
    store = Path(local_store)
    store.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        manifest = artifacts.export_bundle(source, store)
        bundle = artifacts.read_bundle(manifest)
    except Exception as error:
        report = {"status": "failed", "phase": "export", "error": type(error).__name__}
        _record(store, report)
        return [report]
    state_path = store / "sync-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for host in hosts:
        if deadline is not None and time.time() >= deadline:
            break
        report = {
            "host": host,
            "id": bundle["id"],
            "runtime_hash": bundle["runtime_hash"],
        }
        if expected_runtime_hash and bundle["runtime_hash"] != expected_runtime_hash:
            report.update(status="rejected", reason="incompatible_runtime")
        elif state.get(host) == bundle["id"]:
            report.update(status="unchanged")
        else:
            target = (
                Path(remote_store)
                / (".incoming-" + bundle["id"] + "-" + uuid.uuid4().hex)
                / "manifest.json"
            )
            phase = "upload"
            try:
                remote.upload(
                    host,
                    manifest.parent,
                    [*bundle["files"], "manifest.json"],
                    target.parent,
                )
                phase = "validate"
                operation_deadline = (
                    min(deadline, time.time() + 180)
                    if deadline is not None
                    else time.time() + 180
                )
                checked = remote.command(
                    host,
                    [
                        remote.PYTHON,
                        "-c",
                        _PREPARE,
                        str(target),
                        str(operation_deadline),
                        _VALIDATE,
                    ],
                    cwd=code_path,
                    timeout=185,
                )
                receipt = json.loads(checked.strip().splitlines()[-1])
                if (
                    receipt.get("id") != bundle["id"]
                    or receipt.get("runtime_hash") != bundle["runtime_hash"]
                    or receipt.get("hands", 0) < 1
                    or receipt.get("policies") != 4
                ):
                    raise ValueError("validation receipt mismatch")
                phase = "publish"
                remote.command(
                    host,
                    [
                        remote.PYTHON,
                        "-c",
                        _PUBLISH,
                        str(target),
                        bundle["id"],
                        bundle["runtime_hash"],
                        str(operation_deadline),
                        _VALIDATE,
                    ],
                    cwd=code_path,
                    timeout=100,
                )
                state[host] = bundle["id"]
                _atomic_json(state_path, state)
                report.update(
                    status="published",
                    hands=receipt["hands"],
                    policies=receipt["policies"],
                )
            except Exception as error:
                report.update(status="failed", phase=phase, error=type(error).__name__)
                if deadline is None or time.time() < deadline:
                    try:
                        remote.command(
                            host,
                            [
                                remote.PYTHON,
                                "-c",
                                _DISCARD,
                                str(target.parent),
                                str(remote_store),
                            ],
                            cwd=code_path,
                            timeout=10,
                        )
                    except Exception:
                        # An unreachable host can retain an unactivated staging
                        # directory; it must never be mistaken for current.json.
                        report["staging_cleanup"] = "pending"
        reports.append(report)
        if report["status"] != "unchanged":
            _record(store, report)
    return reports


def run_until(
    source,
    local_store,
    remote_store,
    *,
    deadline,
    interval=300,
    expected_runtime_hash=None,
    code_path=remote.ROOT,
):
    if not math.isfinite(deadline) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("finite deadline and positive polling interval required")
    try:
        with _until(deadline):
            while time.time() < deadline:
                sync_once(
                    source,
                    local_store,
                    remote_store,
                    expected_runtime_hash=expected_runtime_hash,
                    deadline=deadline,
                    code_path=code_path,
                )
                remaining = deadline - time.time()
                if remaining > 0:
                    time.sleep(min(interval, remaining))
    except DeadlineReached:
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="projects/agent-research")
    parser.add_argument("--local-store", default=".local/teachers")
    parser.add_argument(
        "--remote-store", default=remote.ROOT + "/.local/guided-teachers"
    )
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--expected-runtime-hash")
    parser.add_argument(
        "--code",
        default=remote.ROOT,
        help="frozen framework checkout used on both remote hosts",
    )
    args = parser.parse_args()
    run_until(
        args.source,
        args.local_store,
        args.remote_store,
        deadline=args.deadline,
        interval=args.interval,
        expected_runtime_hash=args.expected_runtime_hash,
        code_path=args.code,
    )


if __name__ == "__main__":
    main()
