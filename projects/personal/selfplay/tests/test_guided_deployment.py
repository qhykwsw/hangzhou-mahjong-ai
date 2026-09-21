"""Actor artifact isolation and real public-observation HTTP contract checks."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.deployment import (
    BatchService,
    _http_act,
    benchmark,
    create_server,
    export_actor,
    load_actor,
    parse_observation,
    sample_observations,
)
from selfplay_core.actions import action_index


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from guided.networks import Actor

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    actor = Actor(16, 1, 4)
    path = tmp_path / "training.pt"
    torch.save(
        {
            "format": "guided-v1",
            "actor": actor.state_dict(),
            "critic": {"privileged": torch.tensor([999.0])},
            "actor_optimizer": {"training_only": 7},
            "critic_optimizer": {"training_only": 8},
            "replay": ["private-training-cache"],
            "config": {
                "dim": 16,
                "layers": 1,
                "heads": 4,
                "lookahead": False,
                "replay_capacity": 100000,
                "device": "cuda",
            },
        },
        path,
    )
    yield path
    torch.set_num_threads(old_threads)


def test_generated_observations_are_real_and_malformed_or_private_inputs_rejected():
    payloads = sample_observations(8)
    observations = [parse_observation(payload) for payload in payloads]
    assert any(obs.public_events for obs in observations)
    assert all(obs.hand and obs.legal_actions for obs in observations)
    for mutate in (
        lambda x: x.update(legal_actions=[]),
        lambda x: x.update(oracle_planes=[[1]]),
        lambda x: x.update(hand=[99]),
        lambda x: x["stage"].update(seat=-1),
        lambda x: x.update(legal_actions=[{"action": "hu", "tile": 1}]),
    ):
        payload = copy.deepcopy(payloads[0])
        mutate(payload)
        with pytest.raises(ValueError):
            parse_observation(payload)
    incremental = copy.deepcopy(payloads[0])
    incremental.pop("public_events")
    incremental.update(events_from=0, events=[])
    with pytest.raises(ValueError, match="complete"):
        parse_observation(incremental)


def test_export_contains_only_actor_tensors_and_reproduces_weights(
    checkpoint, tmp_path
):
    import torch

    artifact = export_actor(checkpoint, tmp_path / "actor")
    assert {path.name for path in artifact.iterdir()} == {"actor.pt", "meta.json"}
    weights = torch.load(artifact / "actor.pt", weights_only=True)
    assert weights and all(torch.is_tensor(value) for value in weights.values())
    assert not any(
        "critic" in key or "optimizer" in key or "replay" in key for key in weights
    )
    meta = json.loads((artifact / "meta.json").read_text())
    assert set(meta["config"]) == {"dim", "layers", "heads", "lookahead"}
    assert meta["backend"] == "pytorch"
    original, _ = load_actor(checkpoint, "cpu")
    restored, _ = load_actor(artifact, "cpu")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        export_actor(checkpoint, artifact)


def test_tampered_artifact_is_rejected(checkpoint, tmp_path):
    artifact = export_actor(checkpoint, tmp_path / "actor")
    with (artifact / "actor.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_actor(artifact, "cpu")


def test_http_legal_response_and_bad_observation_rejected(checkpoint, tmp_path):
    artifact = export_actor(checkpoint, tmp_path / "actor")
    actor, metadata = load_actor(artifact, "cpu")
    service = BatchService(actor, metadata)
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = sample_observations(1)[0]
        result = _http_act(url, payload)
        obs = parse_observation(payload)
        selected = next(
            action
            for action in obs.legal_actions
            if action_index(action) == result["index"]
        )
        assert json.loads(json.dumps(asdict(selected))) == result["action"]
        broken = copy.deepcopy(payload)
        broken["legal_actions"] = []
        with pytest.raises(urllib.error.HTTPError) as error:
            _http_act(url, broken)
        assert error.value.code == 400
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/health", timeout=5) as response:
            assert json.load(response)["ok"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        service.close()


def test_benchmark_exercises_ten_concurrent_real_observations(checkpoint):
    report = benchmark(checkpoint, count=10, device="cpu", concurrency=10)
    assert report["concurrency"] == 10
    assert report["passed_legality"]
    assert report["gpu"] is None
    assert set(report["loads"]) == {"sustained", "synchronized_burst"}
    for load in report["loads"].values():
        assert load["requests"] == 10
        assert load["illegal_actions"] == load["errors"] == 0
        assert 0 < load["p50_ms"] <= load["p95_ms"] <= load["p99_ms"]


def test_service_bounds_backlog_and_expires_waiting_requests(checkpoint, monkeypatch):
    from concurrent.futures import TimeoutError
    import time
    from guided import encoding
    from guided.deployment import OverloadedError

    actor, metadata = load_actor(checkpoint, "cpu")
    entered, release = threading.Event(), threading.Event()
    original = encoding.encode

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(encoding, "encode", delayed)
    service = BatchService(actor, metadata, max_pending=1, max_batch=1, timeout=0.1)
    try:
        payload = sample_observations(1)[0]
        first = service.submit(payload)
        assert entered.wait(timeout=2)
        waiting = service.submit(payload)
        with pytest.raises(OverloadedError):
            service.submit(payload)
        time.sleep(0.12)
        release.set()
        with pytest.raises(TimeoutError):
            first.result(timeout=2)
        with pytest.raises(TimeoutError):
            waiting.result(timeout=2)
    finally:
        release.set()
        service.close()
