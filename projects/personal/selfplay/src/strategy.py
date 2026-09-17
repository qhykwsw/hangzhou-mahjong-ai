"""Release entrypoint: a frozen ONNX actor served by ONNX Runtime on one CPU thread.

`src/` is copied verbatim into the release as `candidate/`; it may only depend on
numpy, onnxruntime and the shared `mahjong_ai` runtime.  Training code lives in
`../train/` and never enters a release.
"""

from selfplay_core.policy import OnnxPolicy

COMBINATION = "personal-selfplay"
FAMILY = "selfplay"
PARAMETERS = {}
PROCESS_REUSE = True


def build(model_dir):
    return OnnxPolicy(model_dir)
