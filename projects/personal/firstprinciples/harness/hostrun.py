"""Run the rule line's own research scripts (rollouts, data collection, tuning runs) on a GPU host lent to it.

  python -m harness.hostrun push <local dir> --host 4070 [--name <name>]     # copy the directory there (replaces it)
  python -m harness.hostrun run <name> --host 4070 [--sync <path>...] [--models <sha>,...] -- <command...>
  python -m harness.hostrun status <name> --host 4070
  python -m harness.hostrun fetch <name> --host 4070 <file>... [--to <local dir>]
  python -m harness.hostrun stop <name> --host 4070

The command runs detached in the host's copy of the directory, under the host's
Python ("python" as its first word is that interpreter), with:
  FP_REPO        the host's mirror of this repository: a path below the repository
                 here is FP_REPO/<the same relative path> there
  PYTHONPATH     the pool's frozen runtime on the host, compiled there, so
                 mahjong_ai imports as it does in the pool's matches
  MAHJONG_MODEL_SERVER, MAHJONG_SELFPLAY
                 with --models: the host's inference service (CUDA) with those
                 models loaded, for model seats (e.g. the platform proxies)
Directories a script reads below the repository (rule packages, runtimes, model
artifacts, start sets) go up with --sync: each is mirrored once and must not
change afterwards (a changed package needs a new directory, as the attempts'
frozen copies are). A synced rule package's native.c is compiled there before
the command starts. Its output stays in the host's copy until fetched; run.log
holds its stdout and stderr.

The host is lent for the rule line's work (harness.line hosts lend <host> --line rule):
use at most 16 worker processes, the 4070 is also a desktop in daily use.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from . import evaluate as E

sys.path.insert(0, str(E.REPO / "projects/personal/selfplay"))
from mahjong_ai.execution.remote import service_code_files  # noqa: E402

RUNS = "/.local/hostrun/"


def host_of(name):
    return E.remote_host(name)


def target(host, name):
    return host.base + RUNS + name


def push(local, host_name, name=None):
    local = Path(local).resolve()
    name = name or local.name
    host = host_of(host_name)
    where = target(host, name)
    host.python(f"import json,shutil;shutil.rmtree({where!r},ignore_errors=True);print(json.dumps(True))")
    tar = subprocess.run(["tar", "--no-xattrs", "--exclude", "__pycache__", "-czf", "-", "-C", str(local), "."],
                         capture_output=True, env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"}, check=True)
    host._extract(where, tar.stdout)
    print(f"{local} → {host_name}:{where}")


def run(name, host_name, command, sync=(), models=()):
    host = host_of(host_name)
    where = target(host, name)
    runtime = Path(E.load_pool()["runtime_path"])
    paths = [Path(p).resolve() for p in sync]
    host.sync([runtime, *paths, *(E.MODELS / m for m in models)])
    host.build_native([runtime, *(p for p in paths if (p / "mahjong_ai").is_dir())])
    host.build_packages([{"package_path": str(p), "runtime_path": str(runtime)}
                         for p in paths if (p / "native.c").is_file()])
    env = {"FP_REPO": host.base, "PYTHONPATH": host.remote_path(runtime)}
    if models:
        env["MAHJONG_MODEL_SERVER"] = host.ensure_service(E.MODELS, service_code_files())
        env["MAHJONG_SELFPLAY"] = host.service_code + "/projects/personal/selfplay"
        host.warm(models)
    if command and command[0] in ("python", "python3"):
        command = [host.config["python"], *command[1:]]
    script = f"""import json,os,pathlib,subprocess,time
root=pathlib.Path({where!r})
if not root.is_dir(): raise SystemExit('先 push 目录 {name}')
env=dict(os.environ,**{env!r})
log=open(root/'run.log','ab')
p=subprocess.Popen({command!r},cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'run.pid').write_text(str(p.pid))
time.sleep(3)
print(json.dumps({{'pid':p.pid,'alive':p.poll() is None}}))
"""
    started = host.python(script, timeout=600)
    print(f"{host_name}:{where} pid {started['pid']}" + ("" if started["alive"] else "（已退出，看 status）"))
    print("命令：" + shlex.join(command))


def status(name, host_name, lines=20):
    host = host_of(host_name)
    where = target(host, name)
    value = host.python(f"""import json,os,pathlib
root=pathlib.Path({where!r})
pid=int((root/'run.pid').read_text()) if (root/'run.pid').exists() else None
try:
    os.kill(pid,0); alive=True
except Exception:
    alive=False
log=(root/'run.log').read_text(errors='replace').splitlines()[-{lines}:] if (root/'run.log').exists() else []
print(json.dumps({{'pid':pid,'alive':alive,'files':sorted(os.listdir(root)) if root.is_dir() else [],'log':log}}))
""")
    print(f"{host_name}:{where} pid {value['pid']} {'在跑' if value['alive'] else '已结束'}")
    print("文件：" + " ".join(value["files"]))
    print("\n".join(value["log"]))


def fetch(name, host_name, files, to="."):
    host = host_of(host_name)
    out = Path(to)
    out.mkdir(parents=True, exist_ok=True)
    data = host._ssh(f"tar -czf - -C {target(host, name)} " + " ".join(files), timeout=3600)
    subprocess.run(["tar", "-xzf", "-", "-C", str(out)], input=data, check=True)
    print(f"{' '.join(files)} → {out.resolve()}")


def stop(name, host_name):
    host = host_of(host_name)
    where = target(host, name)
    print(host.python(f"""import json,os,pathlib,signal
pid=int(pathlib.Path({where!r},'run.pid').read_text())
try:
    os.killpg(pid,signal.SIGTERM); ok=True
except Exception:
    ok=False
print(json.dumps(ok))
"""))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("push")
    p.add_argument("local")
    p.add_argument("--name")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--sync", nargs="*", default=[])
    r.add_argument("--models", default="")
    for name in ("status", "stop"):
        sub.add_parser(name).add_argument("name")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("files", nargs="+")
    f.add_argument("--to", default=".")
    for s in sub.choices.values():
        s.add_argument("--host", required=True, choices=tuple(h for h in E.EVAL_WORKERS if h != "local"))
    # Everything after "--" is the command, options and all.
    argv = sys.argv[1:]
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if args.command == "push":
        push(args.local, args.host, args.name)
    elif args.command == "run":
        if not command:
            raise SystemExit("run 要在 -- 之后给出命令")
        run(args.name, args.host, command, args.sync, [m for m in args.models.split(",") if m])
    elif args.command == "status":
        status(args.name, args.host)
    elif args.command == "fetch":
        fetch(args.name, args.host, args.files, args.to)
    else:
        stop(args.name, args.host)


if __name__ == "__main__":
    main()
