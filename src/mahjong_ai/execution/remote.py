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
import time
import io

SERVICE_PORT = 8830


class RemoteHost:
    def __init__(self, name, config, project):
        self.name, self.config = name, config
        self.project = Path(project).resolve()
        self.base = config["root"].rstrip("/") + "/.local/research-hosts/" + self.project.name
        self.synced = set()
        self.built = set()
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
        runtimes = [r for r in runtimes if str(r) not in self.built]
        if not runtimes:
            return []
        self.built.update(str(r) for r in runtimes)
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
        # A service left running from older code is replaced: requests and
        # endpoints change with the code (health reports its code root).
        script = f"""import json,os,subprocess,time,urllib.request
url='http://127.0.0.1:{SERVICE_PORT}/health'
def health():
    try: return json.loads(urllib.request.urlopen(url,timeout=2).read())
    except Exception: return {{}}
def ok(): return health().get('ok') is True and health().get('code') == {self.service_code!r}
if health().get('ok') and not ok():
    subprocess.run(['pkill','-f','guided.model_server .*--port {SERVICE_PORT}'])
    for _ in range(40):
        time.sleep(0.5)
        if not health(): break
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

    def warm(self, weights):
        """Load models into the host's service before its matches start."""
        script = f"""import json,urllib.request
for sha in {sorted(weights)!r}:
    r=urllib.request.Request('http://127.0.0.1:{SERVICE_PORT}/load',json.dumps({{'model':sha}}).encode(),{{'Content-Type':'application/json'}})
    assert json.loads(urllib.request.urlopen(r,timeout=600).read()).get('ok'),sha
print(json.dumps(True))
"""
        return self.python(script, timeout=900)

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
        # A snapshot appears only complete, with its compiled extension: a
        # service started on a half-made one would fail every request, and
        # health would still report the right code root.
        if not self.python(f"import json,pathlib;print(json.dumps(pathlib.Path({target!r}).is_dir()))"):
            staging = target + ".tmp"
            self.python(f"import json,shutil;shutil.rmtree({staging!r},ignore_errors=True);print(json.dumps(True))")
            self._extract(staging, buffer.getvalue())
            self.build_native_at(staging + "/src")
            self.python(f"import json,os;os.replace({staging!r},{target!r});print(json.dumps(True))")
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
    SUPERVISOR = """import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); task=json.loads((root/'task.json').read_text())
env=dict(os.environ,**json.loads(sys.argv[2]))
fingerprint=subprocess.run([sys.executable,'-c','from mahjong_ai.persistence.artifacts import runtime_fingerprint;print(runtime_fingerprint())'],env=env,capture_output=True,text=True,check=True).stdout.strip()
task.update(runtime=fingerprint,owner_pid=os.getpid()); (root/'task.json').write_text(json.dumps(task))
with (root/'worker.log').open('ab') as log:
    code=subprocess.call([sys.executable,'-B','-m','mahjong_ai.execution.jobs',str(root)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=str(root))
if not (root/'result.json').exists():
    (root/'result.json').write_text(json.dumps({'status':'failed','error':'remote job exited without a receipt: '+str(code)}))
"""

    def launch_many(self, jobs):
        """Start several jobs, each under its own supervisor, in one round trip.

        ``jobs`` is a list of (local root, task, runtime, env). wsl.exe ends
        the session when the script exits, so the script waits once for the
        detached supervisors to leave it, however many were started.
        """
        entries = []
        for local_root, task, runtime, env in jobs:
            task = dict(task, policies={
                cid: self._interpreter(spec) for cid, spec in task.get("policies", {}).items()
            })
            env = dict(env, PYTHONPATH=self.remote_path(runtime))
            entries.append((self.remote_path(local_root), self._rewrite(task), env))
        payload = base64.b64encode(json.dumps(entries).encode()).decode()
        script = f"""import base64,json,pathlib,subprocess,time
supervisor={self.SUPERVISOR!r}
pids=[]
for root,task,env in json.loads(base64.b64decode({payload!r})):
    root=pathlib.Path(root); root.mkdir(parents=True,exist_ok=True)
    for name in ('result.json','cancel.flag'):
        (root/name).unlink(missing_ok=True)
    (root/'supervisor.py').write_text(supervisor)
    (root/'task.json').write_text(json.dumps(task))
    log=open(root/'supervisor.log','ab')
    p=subprocess.Popen([{self.config['python']!r},str(root/'supervisor.py'),str(root),json.dumps(env)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    pids.append(p.pid)
time.sleep(3)
print(json.dumps(pids))
"""
        return self.python(script, timeout=600)

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

    # Training on a host (imitation, recipe trials) means it is not idle.
    TRAINING = r"guided\.(imitate|trial|run)\b"
    # A host unreachable this long is offline (ADR-0014).
    OFFLINE_SECONDS = 600.0

    def __init__(self, host, models_root, poll_seconds=30.0, probe_seconds=60.0, online=True):
        import queue
        import threading

        self.host, self.models_root, self.poll_seconds = host, Path(models_root), poll_seconds
        self.requests, self.running = queue.Queue(), {}
        self.errors = {}
        self.probe_seconds, self.training = probe_seconds, None  # unknown = busy
        # A host last known offline stays offline until a probe reaches it, so a
        # service restart does not route work to it for OFFLINE_SECONDS.
        self.last_ok = time.monotonic() - (0 if online else self.OFFLINE_SECONDS + 1)
        self.launched = {}  # job id -> wall time of launch, to detect reboots
        # Training process groups suspended while a common brawl runs here.
        self.suspended, self.brawls = [], set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name=f"remote-{host.name}", daemon=True)
        self.thread.start()
        # Reachability is probed on its own thread: a sync of large inputs over
        # a slow relay can occupy the dispatch loop longer than OFFLINE_SECONDS.
        threading.Thread(target=self._watch, name=f"probe-{host.name}", daemon=True).start()

    def offline(self):
        return time.monotonic() - self.last_ok > self.OFFLINE_SECONDS

    def adopt(self, job_id, root):
        """Take over a job launched before a controller restart."""
        marker = Path(root) / "remote.json"
        if not marker.exists():
            return False
        value = json.loads(marker.read_text())
        with self.lock:
            self.running[job_id] = (Path(root), value["pid"])
        self.launched[job_id] = value.get("at", 0.0)
        return True

    def forget(self, job_id):
        """Stop tracking a job that was moved elsewhere; its receipt is ignored."""
        with self.lock:
            self.running.pop(job_id, None)
        self.launched.pop(job_id, None)
        self.brawls.discard(job_id)

    def _watch(self):
        while True:
            self._probe()
            time.sleep(self.probe_seconds)

    def _reachable(self):
        try:
            self.host.python("import json\nprint(json.dumps(True))\n", timeout=60)
        except Exception:
            return False
        self.last_ok = time.monotonic()
        return True

    def _rebooted_since(self, launched):
        """True when the host's Linux started after ``launched`` (its work was lost).

        WSL can stop a distribution without restarting its VM, so the start of
        the distribution's init counts, not the VM uptime.
        """
        try:
            boot = self.host.python(
                "import os\nbtime=[float(l.split()[1]) for l in open('/proc/stat') if l.startswith('btime')][0]\nticks=int(open('/proc/1/stat').read().rsplit(')',1)[1].split()[19])\nprint(btime+ticks/os.sysconf('SC_CLK_TCK'))\n",
                timeout=60,
            )
        except Exception:
            return False
        return boot > launched - 60

    def submit(self, job_id, root, task, runtime, env):
        self.requests.put(("launch", job_id, Path(root), task, runtime, env))

    def cancel(self, job_id, root, brawl=False):
        self.requests.put(("cancel", job_id, Path(root), brawl))

    def call(self, function, root):
        """Run slow host work (syncs, then a hand-off) off the controller loop."""
        self.requests.put(("call", function, None if root is None else Path(root)))

    def busy(self):
        with self.lock:
            return len(self.running) + self.requests.qsize()

    def idle_for_overflow(self):
        """True once a probe saw no training process on the host."""
        return self.training is False

    def _probe(self):
        try:
            self.training = bool(self.host.python(
                "import json,subprocess\nprint(json.dumps(subprocess.run(['pgrep','-f',"
                + repr(self.TRAINING) + "],capture_output=True).returncode==0))\n",
                timeout=120,
            ))
            self.last_ok = time.monotonic()
        except Exception:
            self.training = None

    def _write(self, root, result):
        from mahjong_ai.persistence.artifacts import atomic_json

        atomic_json(root / "result.json", result)

    def _signal_training(self, name):
        """Send ``name`` to every training process group on the host."""
        return self.host.python(
            "import json,os,signal,subprocess\n"
            "pids=[int(p) for p in subprocess.run(['pgrep','-f'," + repr(self.TRAINING)
            + "],capture_output=True,text=True).stdout.split()]\n"
            "groups=set()\n"
            "for p in pids:\n"
            " try: groups.add(os.getpgid(p))\n except OSError: pass\n"
            "for g in groups:\n"
            " try: os.killpg(g,signal." + name + ")\n except OSError: pass\n"
            "print(json.dumps(sorted(groups)))\n",
            timeout=120,
        ) or []

    def _suspend_training(self):
        """A brawl has priority: freeze training (GPU memory kept) until it ends."""
        if not self.suspended:
            self.suspended = self._signal_training("SIGSTOP") or [None]

    def _resume_training(self):
        """Continue whatever training is stopped, including after a controller restart."""
        if not self.brawls:
            try:
                self._signal_training("SIGCONT")
                self.suspended = []
            except Exception:
                pass  # Retried after the next poll.

    def _launch(self, batch):
        """Sync the union of inputs once, then start every queued job together."""
        from mahjong_ai.persistence.artifacts import atomic_json

        try:
            brawl = [job_id for job_id, _, task, _, _ in batch if task.get("scope") == "competition"]
            if brawl:
                self._suspend_training()
                self.brawls.update(brawl)
            inputs, weights = set(), set()
            for _, _, task, _, _ in batch:
                inputs |= job_inputs(task)
                weights |= model_weights(task)
            self.host.sync(inputs | {self.models_root / w for w in weights})
            self.host.build_native(sorted({p for p in inputs if (Path(p) / "mahjong_ai").is_dir()}))
            server = (
                self.host.ensure_service(self.models_root, service_code_files())
                if weights else None
            )
            if weights:
                self.host.warm(weights)
            jobs = []
            for _, root, task, runtime, env in batch:
                env = dict(env)
                if model_weights(task):
                    env["MAHJONG_MODEL_SERVER"] = server
                    env["MAHJONG_SELFPLAY"] = self.host.service_code + "/projects/personal/selfplay"
                jobs.append((root, task, runtime, env))
            pids = self.host.launch_many(jobs)
            self.last_ok = time.monotonic()
        except Exception as exc:
            status = "failed" if self._reachable() else "host_fault"
            for job_id, root, _, _, _ in batch:
                self.brawls.discard(job_id)
                self._write(root, {"status": status, "error": f"远程启动失败：{exc}"[:4000]})
            if self.suspended:
                self._resume_training()
            return
        now = time.time()
        for (job_id, root, _, _, _), pid in zip(batch, pids):
            atomic_json(root / "remote.json", {"host": self.host.name, "pid": pid, "at": now})
            self.launched[job_id] = now
            with self.lock:
                self.running[job_id] = (root, pid)

    def _poll(self):
        with self.lock:
            jobs = dict(self.running)
        if self.suspended:
            self._resume_training()
        if not jobs:
            return
        try:
            states = self.host.poll(jobs)
            self.last_ok = time.monotonic()
        except Exception:
            return  # Unreachable: jobs keep running there; ask again later.
        for job_id, state in states.items():
            root, _ = jobs[job_id]
            with self.lock:
                if job_id not in self.running:
                    continue  # moved elsewhere meanwhile
            if "result" in state:
                self._write(root, state["result"])
            elif not state["alive"]:
                if self._rebooted_since(self.launched.get(job_id, 0.0)):
                    self._write(root, {"status": "host_fault", "error": "远程主机重启，任务丢失"})
                else:
                    self._write(root, {"status": "failed", "error": "远程任务进程已退出且没有回执"})
            else:
                continue
            with self.lock:
                self.running.pop(job_id, None)
            self.launched.pop(job_id, None)
            self.brawls.discard(job_id)
        if self.suspended:
            self._resume_training()

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
                batch = [request[1:]]
                while True:  # Everything queued so far goes out in one round trip.
                    try:
                        more = self.requests.get_nowait()
                    except queue.Empty:
                        break
                    if more[0] == "launch":
                        batch.append(more[1:])
                    else:
                        self.requests.put(more)
                        break
                self._launch(batch)
            elif request and request[0] == "call":
                try:
                    request[1](self.host)
                    self.last_ok = time.monotonic()
                except Exception as exc:
                    if request[2] is not None:
                        status = "failed" if self._reachable() else "host_fault"
                        self._write(request[2], {"status": status, "error": f"远程准备失败：{exc}"[:4000]})
            elif request and request[0] == "cancel":
                with self.lock:
                    entry = self.running.pop(request[1], None)
                self.brawls.discard(request[1])
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
                if request[3]:
                    # A cancelled brawl, possibly from before a restart.
                    self.suspended = self.suspended or [None]
                    self._resume_training()
            if time.monotonic() - last >= self.poll_seconds:
                self._poll()
                last = time.monotonic()
