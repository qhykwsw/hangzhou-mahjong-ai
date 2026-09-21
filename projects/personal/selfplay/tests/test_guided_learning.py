"""Training invariants; torch is an optional, training-host-only dependency."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided.learning import make_batch, masked_log_probs, teacher_cross_entropy, update
from guided.networks import Actor, Critic, PopArt
from selfplay_core.actions import N_ACTIONS
from selfplay_core.encoding import N_PLANES, N_SCALARS


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(17)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def _models():
    return Actor(16, 1, 4), Critic(16, 1, 4)


def _rows(actor, count=4):
    rows = []
    rng = np.random.default_rng(11)
    for i in range(count):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[[2, 5]] = True
        rows.append(
            dict(
                planes=rng.normal(0, 0.1, (N_PLANES, 34)).astype(np.float16),
                scalars=rng.normal(0, 0.1, N_SCALARS).astype(np.float16),
                events=np.asarray([[2, 0, 2, 34, 0]] * (i % 2 + 1), dtype=np.int16),
                mask=mask,
                oracle_planes=np.zeros((16, 34), dtype=np.float32),
                oracle_scalars=np.zeros(4, dtype=np.float32),
                action=(2 if i % 2 else 5),
                teacher=2,
                label_version="teacher-v1",
                value=0.0,
                reward=float(i - 1),
                adv=float(i - 1),
                ret=float(3 * i - 5),
                logprob=0.0,
            )
        )
    batch = make_batch(rows, "cpu")
    with torch.no_grad():
        logp = masked_log_probs(
            actor(batch["planes"], batch["scalars"], batch["events"]), batch["mask"]
        )
        logp = logp.gather(1, batch["action"][:, None]).squeeze(1)
    for row, value in zip(rows, logp.tolist()):
        row["logprob"] = value
    return rows


def test_expired_deadline_prevents_optimizer_steps():
    actor, critic = _models()
    rows = _rows(actor)
    before = copy.deepcopy(actor.state_dict())
    stats = update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(critic.parameters()),
        rows,
        {"deadline": 0},
        1.0,
        [],
        np.random.default_rng(1),
    )
    assert stats["budget_exhausted"] and stats["sgd_steps"] == 0
    for name, value in actor.state_dict().items():
        assert torch.equal(value, before[name])


def test_popart_preserves_raw_output_and_resets_only_head_moments():
    head = PopArt(3, beta=0.5)
    trunk = torch.nn.Linear(3, 3)
    opt = torch.optim.Adam([*trunk.parameters(), *head.parameters()], lr=0.01)
    x = torch.randn(8, 3)
    loss = (head(trunk(x)) - torch.arange(8.0)).square().mean()
    loss.backward()
    opt.step()
    expected = head(trunk(x)).detach().clone()
    trunk_moment = opt.state[trunk.weight]["exp_avg"].clone()
    head.update_stats(torch.tensor([-200.0, 100.0, 500.0]), opt)
    torch.testing.assert_close(head(trunk(x)), expected, atol=3e-5, rtol=3e-5)
    assert head.linear.weight not in opt.state
    assert head.linear.bias not in opt.state
    torch.testing.assert_close(opt.state[trunk.weight]["exp_avg"], trunk_moment)
    # A second very different distribution must preserve the same raw prediction.
    head.update_stats(torch.tensor([1000.0, 2000.0]), opt)
    torch.testing.assert_close(head(trunk(x)), expected, atol=2e-4, rtol=2e-4)


def test_popart_mse_fits_raw_arithmetic_mean_for_stochastic_returns():
    head = PopArt(1)
    targets = torch.tensor([0.0, 100.0])
    head.update_stats(targets)
    opt = torch.optim.SGD(head.parameters(), lr=0.5)
    x = torch.zeros(2, 1)
    loss = (head.normalized(x) - head.normalize(targets)).square().mean()
    loss.backward()
    opt.step()
    torch.testing.assert_close(head(x), torch.full((2,), 50.0))
    assert not torch.isclose(head(x)[0], torch.tensor(101.0**0.5 - 1))


def test_actor_ignores_padding_and_critic_parameters_are_independent():
    actor, critic = _models()
    actor.eval()
    rows = _rows(actor, 2)
    batch = make_batch(rows, "cpu")
    inputs = [batch["planes"], batch["scalars"], batch["events"]]
    with torch.no_grad():
        expected = actor(*inputs)
        padding = torch.full((2, 3, 5), 999, dtype=torch.long)
        padding[..., 0] = 0
        padded = torch.cat([padding, batch["events"], padding], dim=1)
        actual = actor(inputs[0], inputs[1], padded)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert not (
        {p.data_ptr() for p in actor.parameters()}
        & {p.data_ptr() for p in critic.parameters()}
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(1)
        actual = actor(*inputs)
    torch.testing.assert_close(actual, expected)
    assert actor.encoder.positions.shape[0] >= 2049


def test_teacher_cross_entropy_masks_illegal_logits_and_rejects_illegal_label():
    logits = torch.tensor([[1000.0, 1.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[False, True, True]])
    loss = teacher_cross_entropy(logits, mask, torch.tensor([1]))
    torch.testing.assert_close(loss, torch.log1p(torch.exp(torch.tensor(1.0))))
    loss.backward()
    assert logits.grad[0, 0] == 0
    with pytest.raises(ValueError, match="illegal"):
        teacher_cross_entropy(logits, mask, torch.tensor([0]))


def test_joint_update_with_w_zero_ignores_labels_and_replay_but_updates_both_networks():
    actor, critic = _models()
    rows = _rows(actor)
    for row in rows:
        row["teacher"] = -1
        del row["label_version"]
    before_actor = [p.detach().clone() for p in actor.parameters()]
    before_critic = [p.detach().clone() for p in critic.parameters()]
    stats = update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        rows,
        dict(epochs=1, minibatch=2, effective_minibatch=4),
        0,
        [{"malformed": "must never be inspected"}],
        np.random.default_rng(3),
    )
    assert stats["sgd_steps"] == 1
    assert stats["teacher_loss"] == stats["teacher_coefficient"] == 0
    assert stats["replay_fraction"] == 0
    assert any(
        not torch.equal(old, new) for old, new in zip(before_actor, actor.parameters())
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_critic, critic.parameters())
    )
    assert all(np.isfinite(value) for value in stats.values())


def test_critic_only_freezes_actor_and_ignores_labels():
    actor, critic = _models()
    rows = _rows(actor)
    for row in rows:
        row["teacher"] = -1
        del row["label_version"]
    before_actor = [p.detach().clone() for p in actor.parameters()]
    before_critic = [p.detach().clone() for p in critic.parameters()]
    stats = update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        rows,
        dict(epochs=2, minibatch=2, effective_minibatch=2, critic_only=True),
        1.0,
        [{"malformed": "must never be inspected"}],
        np.random.default_rng(3),
    )
    assert stats["critic_only"] and not stats["early_stop"]
    assert stats["sgd_steps"] == 4 and stats["teacher_coefficient"] == 0
    assert all(
        torch.equal(old, new) for old, new in zip(before_actor, actor.parameters())
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_critic, critic.parameters())
    )


def test_bc_only_leaves_critic_and_statistics_untouched_and_improves_teacher_loss():
    actor, critic = _models()
    rows = _rows(actor)
    batch = make_batch(rows, "cpu")
    args = batch["planes"], batch["scalars"], batch["events"]
    before = teacher_cross_entropy(actor(*args), batch["mask"], batch["teacher"]).item()
    critic_state = copy.deepcopy(critic.state_dict())
    critic_opt = torch.optim.Adam(critic.parameters())
    stats = update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        critic_opt,
        rows,
        dict(bc_only=True, epochs=2, minibatch=2, effective_minibatch=4),
        1.0,
        [],
        np.random.default_rng(1),
    )
    after = teacher_cross_entropy(actor(*args), batch["mask"], batch["teacher"]).item()
    assert after < before
    assert stats["value_loss"] == stats["policy_loss"] == 0
    assert not critic_opt.state
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(value, critic_state[name])


def test_microbatch_accumulation_matches_effective_batch_update():
    actor, critic = _models()
    other_actor, other_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    rows = _rows(actor)
    cfg = dict(epochs=1, effective_minibatch=4, target_kl=0.01)
    for a, c, micro in [(actor, critic, 1), (other_actor, other_critic, 4)]:
        update(
            a,
            c,
            torch.optim.SGD(a.parameters(), lr=1e-3),
            torch.optim.SGD(c.parameters(), lr=1e-3),
            rows,
            cfg | {"minibatch": micro},
            0.5,
            [],
            np.random.default_rng(7),
        )
    for p, q in zip(actor.parameters(), other_actor.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    for p, q in zip(critic.parameters(), other_critic.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


def test_kl_stop_discards_pending_gradients_before_parameter_step():
    actor, critic = _models()
    rows = _rows(actor)
    for row in rows:
        row["logprob"] -= 1.0
    expected = copy.deepcopy(actor.state_dict())
    stats = update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(critic.parameters()),
        rows,
        dict(epochs=2, minibatch=2, effective_minibatch=4),
        0,
        [],
        np.random.default_rng(2),
    )
    assert stats["early_stop"] and stats["sgd_steps"] == 0
    assert stats["stop_kl"] > 0.01
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_batch_preserves_label_provenance_and_supports_inference_without_oracle():
    actor, _ = _models()
    rows = _rows(actor, 2)
    rows[1]["label_version"] = "teacher-v2"
    for row in rows:
        del row["oracle_planes"], row["oracle_scalars"]
    batch = make_batch(rows, "cpu")
    assert batch["planes"].dtype == batch["scalars"].dtype == torch.float16
    assert batch["events"].dtype == torch.int64
    assert batch["oracle_planes"].dtype == torch.float32
    assert batch["label_version"] == ["teacher-v1", "teacher-v2"]
    assert not batch["oracle_planes"].any()
    assert batch["events"].shape == (2, 2, 5)


def test_history_teacher_loss_keeps_ninety_ten_weights_and_original_labels():
    actor, critic = _models()
    rows = _rows(actor)
    old = copy.deepcopy(rows[0])
    old["teacher"], old["label_version"] = 5, "teacher-v0"
    batch = make_batch(rows, "cpu")
    previous = make_batch([old], "cpu")
    with torch.no_grad():
        current_ce = teacher_cross_entropy(
            actor(batch["planes"], batch["scalars"], batch["events"]),
            batch["mask"],
            batch["teacher"],
        ).item()
        old_ce = teacher_cross_entropy(
            actor(previous["planes"], previous["scalars"], previous["events"]),
            previous["mask"],
            previous["teacher"],
        ).item()
    stats = update(
        actor,
        critic,
        torch.optim.SGD(actor.parameters(), lr=0.0),
        torch.optim.SGD(critic.parameters(), lr=0.0),
        rows,
        dict(bc_only=True, epochs=1, minibatch=4, effective_minibatch=4),
        1,
        [old],
        np.random.default_rng(3),
    )
    assert stats["teacher_loss"] == pytest.approx(
        0.9 * current_ce + 0.1 * old_ce, abs=2e-6
    )
    assert old["teacher"] == 5 and old["label_version"] == "teacher-v0"
