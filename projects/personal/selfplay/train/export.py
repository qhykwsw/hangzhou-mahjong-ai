"""torch -> ONNX export for the release, with a parity check against torch."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from mahjong_ai.domain.engine import HandState
from mahjong_ai.domain.types import RoundConfig
from selfplay_core.encoding import ENCODING_VERSION, K_EVENTS, N_PLANES, N_SCALARS, encode
from models.common import ActorOnly


def export_onnx(model, cfg, out_dir, version):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = ActorOnly(model).eval().cpu()
    dummy = (
        torch.zeros(1, N_PLANES, 34),
        torch.zeros(1, N_SCALARS),
        torch.zeros(1, K_EVENTS, 5, dtype=torch.int64),
    )
    path = out_dir / "policy.onnx"
    torch.onnx.export(
        actor,
        dummy,
        str(path),
        input_names=["planes", "scalars", "events"],
        output_names=["logits"],
        dynamic_axes={"planes": {0: "b"}, "scalars": {0: "b"}, "events": {0: "b"}, "logits": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "encoding": ENCODING_VERSION,
                "version": version,
                "arch": cfg.arch,
                "lookahead": cfg.lookahead,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=1,
        )
    )
    model.to(cfg.device)
    return path


def sample_observations(count, seed=0):
    rng = random.Random(seed)
    out = []
    hand = HandState(RoundConfig(), seed)
    while len(out) < count:
        if hand.settled:
            hand = HandState(RoundConfig(you_cai_bi_kao=rng.random() < 0.5), rng.randrange(1 << 30))
        legal = hand.legal_actions()
        if len(legal) > 1:
            out.append(hand.observation())
        hand.step(rng.choice(legal))
    return out


def verify_onnx(model, out_dir, cfg, count=64, atol=1e-3):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(Path(out_dir) / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    model.eval()
    worst = 0.0
    agree = 0
    device = torch.device("cpu")  # fp32 on CPU: no TF32/bf16 in the reference path
    original_device = next(model.parameters()).device
    model.to(device)
    for obs in sample_observations(count, seed=123):
        enc = encode(obs, lookahead=cfg.lookahead)
        (ort_logits,) = session.run(["logits"], {"planes": enc.planes[None], "scalars": enc.scalars[None], "events": enc.events[None].astype(np.int64)})
        with torch.no_grad():
            t_logits = model.actor_logits(
                torch.from_numpy(enc.planes[None]).to(device),
                torch.from_numpy(enc.scalars[None]).to(device),
                torch.from_numpy(enc.events[None].astype(np.int64)).to(device),
            ).cpu().numpy()
        worst = max(worst, float(np.abs(ort_logits - t_logits).max()))
        agree += int(np.argmax(ort_logits[0][enc.legal]) == np.argmax(t_logits[0][enc.legal]))
    model.to(original_device)
    return {"max_abs_diff": worst, "greedy_agreement": agree / count, "ok": worst < atol and agree == count}
