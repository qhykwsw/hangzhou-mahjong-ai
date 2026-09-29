"""A package's own C (native.c beside its strategy) builds once per machine and loads as its own module."""

import pytest

fast = pytest.importorskip("mahjong_ai._fast", reason="C accelerator not built")

PLAIN = """
#include <Python.h>
static PyObject *which(PyObject *self, PyObject *args) { return PyUnicode_FromString("%s"); }
static PyMethodDef methods[] = {{"which", which, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, methods};
PyMODINIT_FUNC PyInit_native(void) { return PyModule_Create(&module); }
"""

WITH_FAST = """
#include "_fast.c"
static PyObject *distance(PyObject *self, PyObject *arg) {
    int c[NTILES];
    if (parse_counts(arg, c) < 0) return NULL;
    return PyLong_FromLong(shanten_raw(c, 0));
}
static PyMethodDef methods[] = {{"distance", distance, METH_O, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "native", NULL, -1, methods};
PyMODINIT_FUNC PyInit_native(void) { return fast_keys() < 0 ? NULL : PyModule_Create(&module); }
"""


def package(root, name, source):
    folder = root / name
    folder.mkdir()
    (folder / "native.c").write_text(source)
    (folder / "strategy.py").write_text("")
    return folder / "strategy.py"


def test_each_package_gets_its_own_native_module_built_once(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHJONG_NATIVE_CACHE", str(tmp_path / "cache"))
    a = fast.load_native(str(package(tmp_path, "a", PLAIN % "a")))
    b = fast.load_native(str(package(tmp_path, "b", PLAIN % "b")))
    assert (a.which(), b.which()) == ("a", "b")
    assert fast.load_native(str(tmp_path / "a" / "strategy.py")) is a
    assert len(list((tmp_path / "cache").glob("*.so"))) == 2  # outside the packages: their hashes stay


def test_native_code_can_use_the_runtimes_primitives(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHJONG_NATIVE_CACHE", str(tmp_path / "cache"))
    native = fast.load_native(str(package(tmp_path, "c", WITH_FAST)))
    hand = [0] * 34
    for tile in (0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 30, 31):
        hand[tile] += 1
    assert native.distance(hand) == fast.shanten_raw(hand, 0)


def test_a_build_failure_names_the_packages_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHJONG_NATIVE_CACHE", str(tmp_path / "cache"))
    strategy = package(tmp_path, "bad", "int f( { return 0; }\n")
    with pytest.raises(RuntimeError, match="native.c does not build") as failure:
        fast.load_native(str(strategy))
    assert str(tmp_path / "bad" / "native.c") in str(failure.value)


def test_a_host_builds_packages_own_c_before_jobs_load_them(tmp_path, monkeypatch):
    """Training jobs count their processes: the compiler must not run inside one (RemoteHost.build_packages)."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    from mahjong_ai.execution.remote import RemoteHost

    monkeypatch.setenv("MAHJONG_NATIVE_CACHE", str(tmp_path / "cache"))
    runtime = Path(fast.__file__).resolve().parents[1]
    with_c = package(tmp_path, "with_c", PLAIN % "x").parent
    without = tmp_path / "plain"
    without.mkdir()
    (without / "strategy.py").write_text("")
    host = RemoteHost("h", {"root": "/r", "ssh": "x", "distro": "d", "user": "u", "python": "py"}, tmp_path)
    host.remote_path = lambda local: str(local)  # the host is this machine

    def run(script, timeout=None):
        out = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    host.python = run
    specs = [{"package_path": str(with_c), "runtime_path": str(runtime)},
             {"package_path": str(without), "runtime_path": str(runtime)}]
    assert host.build_packages(specs) == [str(with_c)]
    assert len(list((tmp_path / "cache").glob("*.so"))) == 1
    assert host.build_packages(specs) == []  # once per host
