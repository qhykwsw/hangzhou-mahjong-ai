"""Mac control plane for the Windows/WSL training hosts.

Commands use argv through Python, never interpolated shell commands. SSH uses
existing host keys. Only explicitly enumerated source/artifact files transfer.

How to reach each host is machine-specific and stays out of Git: it is read from
the repository's `.local/training-hosts.json`, laid out like
`training-hosts.example.json` beside this file (which stands in, with
placeholder addresses, when there is none).
"""

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

CONFIG = Path(__file__).resolve().parents[4] / ".local/training-hosts.json"
EXAMPLE = Path(__file__).with_name("training-hosts.example.json")


def _load_profiles():
    config = json.loads((CONFIG if CONFIG.is_file() else EXAMPLE).read_text())
    defaults = {"root": config["root"], "python": config["python"]}
    return config["root"], config["python"], {
        name: {**defaults, **profile} for name, profile in config["hosts"].items()
    }


ROOT, PYTHON, PROFILES = _load_profiles()
HOSTS = {name: profile["ssh"] for name, profile in PROFILES.items()}


def root(host):
    return PROFILES[host]["root"]


def python(host):
    return PROFILES[host]["python"]


def wsl(host):
    profile = PROFILES[host]
    return f"wsl.exe -d {profile['distro']} -u {profile['user']} -- "


def execute(host, script, *, timeout=120):
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=8",
        HOSTS[host],
        wsl(host) + python(host) + " -",
    ]
    # wsl.exe now and then prints a Windows system message in the console's code
    # page (GBK here): decoding must not fail the caller over a garbled line.
    result = subprocess.run(
        args, input=script, capture_output=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"{host}: {result.stdout}\n{result.stderr}")
    return result.stdout


def command(
    host, argv, *, cwd=None, environment=None, background_log=None, timeout=120
):
    task = {
        "argv": list(map(str, argv)),
        "cwd": str(cwd or root(host)),
        "env": environment or {},
        "log": background_log,
    }
    script = "import subprocess,os,json,pathlib\ntask=" + repr(task) + "\n"
    script += "env=dict(os.environ,**task['env'])\n"
    script += (
        "overlay=pathlib.Path(" + repr(root(host) + "/.local/guided-deps/numpy-2.1.2") + ")\n"
    )
    script += "if overlay.is_dir(): env['PYTHONPATH']=str(overlay)+os.pathsep+env.get('PYTHONPATH','')\n"
    if background_log:
        script += """path=pathlib.Path(task['log']);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('ab') as stream:
 p=subprocess.Popen(task['argv'],cwd=task['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
# wsl.exe tears down the session when this script exits; give the detached
# child time to leave it, or it can be killed before writing a single line.
import time;time.sleep(5)
print(json.dumps({'pid':p.pid,'log':str(path)}))
"""
    else:
        script += "subprocess.run(task['argv'],cwd=task['cwd'],env=env,check=True)\n"
    return execute(host, script, timeout=timeout)


def upload(host, root, files, destination):
    root = Path(root).resolve()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name in sorted(set(map(str, files))):
            path = root / name
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(root)
            ):
                raise ValueError(f"not a regular source file: {name}")
            archive.add(path, arcname=name, recursive=False)
    payload = base64.b64encode(buf.getvalue()).decode()
    script = (
        "import pathlib,tarfile,io,base64,os\nroot=pathlib.Path("
        + repr(str(destination))
        + ")\n"
    )
    script += "root.mkdir(parents=True,exist_ok=True)\n"
    script += "data=base64.b64decode(" + repr(payload) + ")\n"
    script += """with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=root/member.name
  if not member.isfile() or os.path.commonpath([str(target.resolve()),str(root.resolve())]) != str(root.resolve()):
   raise ValueError('invalid transfer member')
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as source, target.open('wb') as dest:
   dest.write(source.read())
 print('uploaded',len(archive.getmembers()),'files to',str(root))
"""
    return execute(host, script, timeout=180)


def source_files(repo):
    manifest = Path(repo) / "source-manifest.json"
    if manifest.is_file():
        # Frozen control planes need no Git checkout, and must not discover
        # files from an enclosing live repository.
        import json

        return sorted(json.loads(manifest.read_text())["files"])
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return [name.decode() for name in result.stdout.split(b"\0") if name]


def download(host, source, destination):
    """Copy an explicitly selected experiment artifact with transport verification."""
    script = (
        "import pathlib,base64,json,hashlib\ndata=pathlib.Path("
        + repr(str(source))
        + ").read_bytes()\n"
    )
    script += "print(json.dumps({'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}))\n"
    result = json.loads(execute(host, script, timeout=180))
    data = base64.b64decode(result["data"], validate=True)
    if hashlib.sha256(data).hexdigest() != result["sha256"]:
        raise ValueError("artifact transfer checksum mismatch")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path
