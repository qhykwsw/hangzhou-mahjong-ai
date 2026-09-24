import numpy as np
import pytest

torch = pytest.importorskip("torch")
from guided.inference import (
    policy_batch,
    evaluate_requests,
    policy_logits,
    prepare_frozen,
)
from guided.learning import make_batch
from guided.networks import Actor
from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


def rows():
    rng = np.random.default_rng(91)
    result = []
    for length in [0, 1, 7, 3]:
        mask = rng.random(N_ACTIONS) > 0.7
        event = np.zeros((length, 5), dtype=np.int64)
        if length:
            event[:, 0] = 1
        result.append(
            dict(
                planes=rng.normal(size=(N_PLANES, 34)).astype(np.float16),
                scalars=rng.normal(size=N_SCALARS).astype(np.float16),
                events=event,
                mask=mask,
            )
        )
    return result


def test_actor_only_batch_is_exactly_equivalent_without_training_fields():
    inputs = rows()
    full = make_batch(inputs, "cpu")
    actual = policy_batch(inputs, "cpu")
    assert set(actual) == {"planes", "scalars", "events", "mask"}
    for key, value in actual.items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=0)
    inputs[0]["mask"][:] = False
    with pytest.raises(ValueError, match="legal"):
        policy_batch(inputs, "cpu")


def test_precast_frozen_model_supports_eager_inference_without_graphs():
    import copy

    torch.set_num_threads(1)
    torch.manual_seed(93)
    actor = Actor(16, 1, 4).eval()
    # Exercise fp16 coefficients on the CPU test runner, including the eager
    # fallback used when a training recipe installs a custom sampling hook.
    frozen = prepare_frozen(copy.deepcopy(actor), "cuda").cpu()
    assert actor.encoder.tile_in.weight.dtype == torch.float32
    assert frozen.encoder.norm.weight.dtype == torch.float32
    b = policy_batch(rows(), "cpu")
    args = [b[k] for k in ("planes", "scalars", "events")]
    with torch.no_grad(), torch.autocast("cpu", torch.float16, cache_enabled=False):
        expected = frozen(*args)
        reference = actor(*args)
    with torch.no_grad():
        actual = policy_logits(frozen, *args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The CPU fallback may select a different fused bias kernel for precast
    # parameters; its roundoff stays within fp16 precision.
    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=2e-3)


@pytest.mark.parametrize("max_batch", [1, 128])
def test_mixed_models_and_connections_keep_reference_decisions(max_batch):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    models = {
        "model:large": Actor(24, 2, 4).eval(),
        "model:small": Actor(16, 1, 4).eval(),
    }
    inputs = rows()
    pending = []
    expected = {}
    for i, row in enumerate(inputs):
        slot = "model:large" if i % 2 else "model:small"
        pending.append((f"conn-{i % 2}", i, {"row": row, "slot": slot}))
    for slot, model in models.items():
        group = [p for p in pending if p[2]["slot"] == slot]
        b = make_batch([p[2]["row"] for p in group], "cpu")
        with torch.no_grad():
            actions = (
                model(b["planes"], b["scalars"], b["events"])
                .float()
                .masked_fill(~b["mask"], -1e9)
                .log_softmax(-1)
                .argmax(-1)
                .tolist()
            )
        expected.update({(p[0], p[1]): a for p, a in zip(group, actions)})
    result, sizes = evaluate_requests(
        pending, models["model:small"], models, "cpu", max_batch=max_batch
    )
    assert {(c, i): r[0] for c, i, r in result} == expected
    assert sum(sizes) == len(inputs)
    assert evaluate_requests([], models["model:small"], models, "cpu") == ([], [])


@pytest.mark.parametrize("layers", [1, 3])
def test_batch_first_attention_matches_stock_outputs_and_gradients(layers):
    import copy

    torch.set_num_threads(1)
    torch.manual_seed(250925)
    actual = Actor(24, layers, 4).eval()
    reference = copy.deepcopy(actual)
    stock = torch.nn.TransformerEncoderLayer(
        24, 4, 96, dropout=0, activation="gelu", batch_first=True, norm_first=True
    )
    reference.encoder.transformer = torch.nn.TransformerEncoder(
        stock, layers, enable_nested_tensor=False
    ).eval()
    reference.load_state_dict(actual.state_dict(), strict=True)
    b = policy_batch(rows(), "cpu")
    inputs = [b[k] for k in ("planes", "scalars", "events")]
    before, after = reference(*inputs), actual(*inputs)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-5)
    # A real masked policy loss propagates through history keys/values even
    # though the last block no longer builds unused history output queries.
    target = torch.randn_like(before)
    (before.masked_fill(~b["mask"], 0) * target).sum().backward()
    (after.masked_fill(~b["mask"], 0) * target).sum().backward()
    for (name, a), (_, r) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(a.grad, r.grad, atol=2e-5, rtol=2e-4, msg=name)
