"""Run match jobs on a configured GPU host and fetch their receipts.

The control plane stays the only owner of research state. A remote job gets
exactly the immutable inputs it names (frozen runtimes, candidate packages,
model artifacts), mirrored below the host's base directory at the same path
relative to the project root, and writes its receipt there; the controller
polls and copies the receipt back, then accepts it like a local one.

Hosts are Windows machines reached over SSH that run jobs inside WSL
(``wsl.exe -d <distro> -u <user>``), as configured under ``[hosts.<name>]``.
"""

import base64
import json
from pathlib import Path
import subprocess
import tarfile
import io

SERVICE_PORT = 8830


class RemoteHost:
    def __init__(self, name, config, project):
        self.name, self.config = name, config
        self.project = Path(project).resolve()
        self.base = config["root"].rstrip("/") + "/.local/research-hosts/" + self.project.name
        self.synced = set()
        self.service_code = None

    # -- transport ---------------------------------------------------------
    def _ssh(self, command, *, input=None, timeout=300):
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.config["ssh"],
                f"wsl.exe -d {self.config['distro']} -u {self.config['user']} -- {command}"]
        result = subprocess.run(args, input=input, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{self.name}: {result.stderr.decode(errors='replace')[-2000:]}")
        return result.stdout

    def python(self, script, timeout=300):
        """Run a Python script on the host; its last stdout line is JSON."""
        out = self._ssh(self.config["python"] + " -", input=script.encode(), timeout=timeout)
        lines = out.decode(errors="replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else None

    def _marker(self, local):
        relative = Path(local).resolve().relative_to(self.project).as_posix()
        return self.base + "/.synced/" + relative + ".ok"

    def remote_path(self, local):
        local = Path(local).resolve()
        if not local.is_relative_to(self.project):
            raise ValueError(f"remote jobs only use files below the project: {local}")
        return self.base + "/" + local.relative_to(self.project).as_posix()

    # -- inputs ------------------------------------------------------------
    def sync(self, paths):
        """Mirror immutable directories that the host does not have yet."""
        wanted = sorted({Path(p).resolve() for p in paths} - self.synced)
        if not wanted:
            return
        present = self.python(
            "import json,os\nprint(json.dumps([os.path.isfile(p) for p in "
            + repr([self._marker(p) for p in wanted]) + "]))\n"
        )
        missing = [p for p, ok in zip(wanted, present) if not ok]
        for path in missing:
            relative = path.relative_to(self.project).as_posix()
            tar = subprocess.Popen(
                ["tar", "--no-xattrs", "--exclude", "__pycache__", "-czf", "-", "-C",
                 str(self.project), relative],
                stdout=subprocess.PIPE,
                env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"},
            )
            data = tar.communicate()[0]
            if tar.returncode:
                raise RuntimeError(f"cannot archive {path}")
            self._extract(self.base, data)
            # Markers live outside the mirrored tree: packages are verified
            # by content hash, so nothing may be added inside them.
            self.python(
                f"import json,pathlib\nm=pathlib.Path({self._marker(path)!r})"
                "\nm.parent.mkdir(parents=True,exist_ok=True);m.touch()\nprint(json.dumps(True))\n"
            )
        self.synced.update(wanted)

    def _extract(self, target, data):
        # cmd.exe and wsl.exe do not honour shell quoting: create the directory
        # from Python and run tar as a plain argument list.
        self.python(
            f"import json,pathlib\npathlib.Path({target!r}).mkdir(parents=True,exist_ok=True)"
            "\nprint(json.dumps(True))\n"
        )
        self._ssh(f"tar -xzf - -C {target}", input=data, timeout=3600)

    def build_native(self, runtimes):
        """Compile the engine accelerator inside mirrored runtimes on this host."""
        script = (
            "import json,shutil,sysconfig,tempfile\nfrom pathlib import Path\n"
            "from setuptools import Extension,Distribution\nbuilt=[]\n"
            "for root in " + repr([self.remote_path(r) for r in runtimes]) + ":\n"
            " src=Path(root)/'mahjong_ai';target=src/('_fast'+sysconfig.get_config_var('EXT_SUFFIX'))\n"
            " if target.exists() or not (src/'_fast.c').exists(): continue\n"
            " with tempfile.TemporaryDirectory() as f:\n"
            "  d=Distribution({'ext_modules':[Extension('mahjong_ai._fast',sources=[str(src/'_fast.c')],extra_compile_args=['-O3'])]})\n"
            "  c=d.get_command_obj('build_ext');c.build_lib=f+'/l';c.build_temp=f+'/t';d.run_command('build_ext')\n"
            "  shutil.copyfile(c.get_ext_fullpath('mahjong_ai._fast'),target);built.append(root)\n"
            "print(json.dumps(built))\n"
        )
        return self.python(script, timeout=900)

    # -- model inference ---------------------------------------------------
    def ensure_service(self, models_root, code_files):
        """Start (once) the host's inference service over the mirrored models."""
        if self.service_code is None:
            self.service_code = self._upload_code(code_files)
        script = f"""import json,os,subprocess,time,urllib.request
url='http://127.0.0.1:{SERVICE_PORT}/health'
def ok():
    try: return json.loads(urllib.request.urlopen(url,timeout=2).read()).get('ok') is True
    except Exception: return False
if not ok():
    log=open({self.service_code + '/model-server.log'!r},'ab')
    env=dict(os.environ); [env.pop(k,None) for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY')]
    subprocess.Popen([{self.config['python']!r},'-m','guided.model_server','--root',{self.remote_path(models_root)!r},'--port','{SERVICE_PORT}'],
        cwd={self.service_code + '/projects/personal/selfplay'!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(240):
        time.sleep(0.5)
        if ok(): break
print(json.dumps(ok()))
"""
        if not self.python(script, timeout=300):
            raise RuntimeError(f"{self.name}: 模型推理服务未能启动")
        return f"http://127.0.0.1:{SERVICE_PORT}"

    def _upload_code(self, files):
        import hashlib

        repo = Path(__file__).resolve().parents[3]
        digest = hashlib.sha256()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name in sorted(files):
                data = (repo / name).read_bytes()
                digest.update(name.encode() + b"\0" + data)
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        target = self.base + "/service-code/" + digest.hexdigest()[:16]
        self._extract(target, buffer.getvalue())
        self.build_native_at(target + "/src")
        return target

    def build_native_at(self, source_root):
        script = (
            "import json,shutil,sysconfig,tempfile\nfrom pathlib import Path\n"
            "from setuptools import Extension,Distribution\n"
            f"src=Path({source_root!r})/'mahjong_ai';target=src/('_fast'+sysconfig.get_config_var('EXT_SUFFIX'))\n"
            "if not target.exists():\n"
            " with tempfile.TemporaryDirectory() as f:\n"
            "  d=Distribution({'ext_modules':[Extension('mahjong_ai._fast',sources=[str(src/'_fast.c')],extra_compile_args=['-O3'])]})\n"
            "  c=d.get_command_obj('build_ext');c.build_lib=f+'/l';c.build_temp=f+'/t';d.run_command('build_ext')\n"
            "  shutil.copyfile(c.get_ext_fullpath('mahjong_ai._fast'),target)\n"
            "print(json.dumps(str(target)))\n"
        )
        return self.python(script, timeout=900)

    # -- jobs --------------------------------------------------------------
    def launch(self, local_root, task, runtime, env):
        """Start one job under a supervisor that owns it on the host."""
        remote_root = self.remote_path(local_root)
        task = dict(task, policies={
            cid: self._interpreter(spec) for cid, spec in task.get("policies", {}).items()
        })
        payload = base64.b64encode(json.dumps(self._rewrite(task)).encode()).decode()
        script = f"""import base64,json,os,pathlib,subprocess,sys,time
root=pathlib.Path({remote_root!r}); root.mkdir(parents=True,exist_ok=True)
for name in ('result.json','cancel.flag'):
    (root/name).unlink(missing_ok=True)
task=json.loads(base64.b64decode({payload!r}))
supervisor='''import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); task=json.loads((root/'task.json').read_text())
env=dict(os.environ,**json.loads(sys.argv[2]))
fingerprint=subprocess.run([sys.executable,'-c','from mahjong_ai.persistence.artifacts import runtime_fingerprint;print(runtime_fingerprint())'],env=env,capture_output=True,text=True,check=True).stdout.strip()
task.update(runtime=fingerprint,owner_pid=os.getpid()); (root/'task.json').write_text(json.dumps(task))
with (root/'worker.log').open('ab') as log:
    code=subprocess.call([sys.executable,'-B','-m','mahjong_ai.execution.jobs',str(root)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=str(root))
if not (root/'result.json').exists():
    (root/'result.json').write_text(json.dumps({{'status':'failed','error':'remote job exited without a receipt: '+str(code)}}))
'''
(root/'supervisor.py').write_text(supervisor)
(root/'task.json').write_text(json.dumps(task))
environment=json.loads({json.dumps(env)!r}); environment['PYTHONPATH']={self.remote_path(runtime)!r}
log=open(root/'supervisor.log','ab')
p=subprocess.Popen([{self.config['python']!r},str(root/'supervisor.py'),str(root),json.dumps(environment)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(3)
print(json.dumps({{'pid':p.pid}}))
"""
        return self.python(script, timeout=300)["pid"]

    def _interpreter(self, spec):
        """Candidates run under the host's interpreter; locked venvs stay local."""
        python = spec.get("python_path")
        if not python:
            return spec
        if Path(python).resolve().is_relative_to(self.project):
            raise ValueError("带独立依赖环境的候选只能在本机评估")
        return dict(spec, python_path=self.config["python"])

    def poll(self, jobs):
        """Receipts of finished jobs, and whether each unfinished one is alive.

        ``jobs`` maps job id -> (local root, supervisor pid).
        """
        request = {jid: (self.remote_path(root), pid) for jid, (root, pid) in jobs.items()}
        script = f"""import json,os,pathlib
out={{}}
for jid,(root,pid) in {request!r}.items():
    r=pathlib.Path(root)/'result.json'
    if r.exists():
        out[jid]={{'result':json.loads(r.read_text())}}
    else:
        try: os.kill(pid,0); alive=True
        except OSError: alive=False
        out[jid]={{'alive':alive}}
print(json.dumps(out))
"""
        return self.python(script, timeout=300)

    def cancel(self, local_root, pid):
        script = f"""import json,os,pathlib,signal
pathlib.Path({self.remote_path(local_root)!r},'cancel.flag').touch()
try: os.killpg({int(pid)},signal.SIGTERM)
except OSError: pass
print(json.dumps(True))
"""
        return self.python(script, timeout=120)

    def _rewrite(self, value):
        prefix = str(self.project)
        if isinstance(value, str) and (value == prefix or value.startswith(prefix + "/")):
            return self.remote_path(value)
        if isinstance(value, dict):
            return {k: self._rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._rewrite(v) for v in value]
        return value


def opponent_specs(host, candidates, models_root, timeouts):
    """Host-side specs of frozen versions for training seats, synced first."""
    specs, paths, runtimes = {}, set(), set()
    for cid, candidate in candidates.items():
        spec = candidate["spec"]
        if spec.get("family") == "model":
            local = Path(models_root) / spec["parameters"]["weights"]
            paths.add(local)
            specs[cid] = {"family": "model", "path": host.remote_path(local)}
            continue
        spec = dict(spec, action_seconds=timeouts["discard"], response_seconds=timeouts["response"])
        paths |= {spec["package_path"], spec["runtime_path"]}
        runtimes.add(spec["runtime_path"])
        specs[cid] = host._rewrite(host._interpreter(spec))
    host.sync(paths)
    host.build_native(sorted(runtimes))
    return specs


def service_code_files():
    """Repository files the host's inference service needs."""
    repo = Path(__file__).resolve().parents[3]
    files = subprocess.run(
        ["git", "ls-files", "src/mahjong_ai", "projects/personal/selfplay/guided",
         "projects/personal/selfplay/src"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [f for f in files if (repo / f).is_file()]


def job_inputs(task):
    """Immutable directories a match task reads."""
    paths = {task["runtime_path"]}
    for spec in task.get("policies", {}).values():
        for key in ("package_path", "runtime_path"):
            if spec.get(key):
                paths.add(spec[key])
    return paths


def model_weights(task):
    return {
        spec["parameters"]["weights"]
        for spec in task.get("policies", {}).values()
        if spec.get("family") == "model"
    }




class Dispatcher:
    """Background launcher and poller for one host.

    SSH round trips and first-time syncs can take minutes over a relay, so
    they never run under the research store's write lock: the controller
    only hands jobs over, and receipts appear in the local job directory
    exactly as a local worker would have written them.
    """

    def __init__(self, host, models_root, poll_seconds=30.0):
        import queue
        import threading

        self.host, self.models_root, self.poll_seconds = host, Path(models_root), poll_seconds
        self.requests, self.running = queue.Queue(), {}
        self.errors = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name=f"remote-{host.name}", daemon=True)
        self.thread.start()

    def submit(self, job_id, root, task, runtime, env):
        self.requests.put(("launch", job_id, Path(root), task, runtime, env))

    def cancel(self, job_id, root):
        self.requests.put(("cancel", job_id, Path(root)))

    def call(self, function, root):
        """Run slow host work (syncs, then a hand-off) off the controller loop."""
        self.requests.put(("call", function, Path(root)))

    def busy(self):
        with self.lock:
            return len(self.running) + self.requests.qsize()

    def _write(self, root, result):
        from mahjong_ai.persistence.artifacts import atomic_json

        atomic_json(root / "result.json", result)

    def _launch(self, job_id, root, task, runtime, env):
        try:
            inputs = job_inputs(task)
            weights = model_weights(task)
            self.host.sync(inputs | {self.models_root / w for w in weights})
            self.host.build_native(sorted({p for p in inputs if (Path(p) / "mahjong_ai").is_dir()}))
            env = dict(env)
            if weights:
                env["MAHJONG_MODEL_SERVER"] = self.host.ensure_service(
                    self.models_root, service_code_files()
                )
            pid = self.host.launch(root, task, runtime, env)
            from mahjong_ai.persistence.artifacts import atomic_json

            atomic_json(root / "remote.json", {"host": self.host.name, "pid": pid})
            with self.lock:
                self.running[job_id] = (root, pid)
        except Exception as exc:
            self._write(root, {"status": "failed", "error": f"远程启动失败：{exc}"[:4000]})

    def _poll(self):
        with self.lock:
            jobs = dict(self.running)
        if not jobs:
            return
        try:
            states = self.host.poll(jobs)
        except Exception:
            return  # A relay hiccup: try again on the next poll.
        for job_id, state in states.items():
            root, _ = jobs[job_id]
            if "result" in state:
                self._write(root, state["result"])
            elif not state["alive"]:
                self._write(root, {"status": "failed", "error": "远程任务进程已退出且没有回执"})
            else:
                continue
            with self.lock:
                self.running.pop(job_id, None)

    def _loop(self):
        import queue
        import time

        last = 0.0
        while True:
            try:
                request = self.requests.get(timeout=1.0)
            except queue.Empty:
                request = None
            if request and request[0] == "launch":
                self._launch(*request[1:])
            elif request and request[0] == "call":
                try:
                    request[1](self.host)
                except Exception as exc:
                    self._write(request[2], {"status": "failed", "error": f"远程准备失败：{exc}"[:4000]})
            elif request and request[0] == "cancel":
                with self.lock:
                    entry = self.running.pop(request[1], None)
                marker = request[2] / "remote.json"
                if entry is None and marker.exists():
                    # Launched before a controller restart.
                    entry = (request[2], json.loads(marker.read_text())["pid"])
                if entry:
                    try:
                        self.host.cancel(*entry)
                    except Exception:
                        pass
                    self._write(entry[0], {"status": "paused", "reason": "operator interruption"})
            if time.monotonic() - last >= self.poll_seconds:
                self._poll()
                last = time.monotonic()
