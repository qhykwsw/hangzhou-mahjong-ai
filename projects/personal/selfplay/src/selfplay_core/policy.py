"""ONNX Runtime actor for the frozen release: CPU, one thread, greedy."""

import json
from pathlib import Path

import numpy as np

from .encoding import ENCODING_VERSION, encode


class OnnxPolicy:
    name = "selfplay"
    version = "selfplay-onnx"

    def __init__(self, model_dir):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        if meta.get("encoding") != ENCODING_VERSION:
            raise ValueError(
                f"model encoding {meta.get('encoding')!r} != {ENCODING_VERSION!r}"
            )
        self.version = str(meta.get("version", self.version))
        self.lookahead = bool(meta.get("lookahead", True))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def reset(self, seed):
        pass

    def finish_round(self, result, seat):
        pass

    def logits(self, obs):
        enc = encode(obs, lookahead=self.lookahead)
        (out,) = self.session.run(
            ["logits"],
            {
                "planes": enc.planes[None],
                "scalars": enc.scalars[None],
                "events": enc.events[None].astype(np.int64),
            },
        )
        return out[0], enc

    def act(self, obs):
        if not obs.legal_actions:
            raise ValueError("no legal actions")
        logits, enc = self.logits(obs)
        scores = logits[enc.legal]
        return obs.legal_actions[int(np.argmax(scores))]
