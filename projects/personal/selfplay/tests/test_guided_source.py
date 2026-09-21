import hashlib
import json

import pytest

from guided import bootstrap


def test_deployed_source_manifest_detects_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "REPO", tmp_path)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    files = {"module.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "source-manifest.json").write_text(
        json.dumps({"id": identity, "files": files})
    )
    assert bootstrap.verify_source() == identity
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        bootstrap.verify_source()
