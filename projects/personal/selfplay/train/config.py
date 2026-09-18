"""Training configuration: dataclass defaults, TOML overrides, run directories."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]


@dataclass
class Config:
    # ---- identity ----
    run_name: str = "dev"
    arch: str = "resnet"  # mlp | resnet | transformer
    seed: int = 1
    # ---- environment / episodes ----
    # Live tournament config (t_65d538e905c5, 2026-09-17): Rounds 16, BaseScore 1,
    # YouCaiBiKao off.  Most games use the live hand count; a share keeps the
    # policy general across other lengths.
    # 正式赛各阶段的局数以 online-probe 读到的配置为准；二测海选实测 16 局。
    rounds_choices: tuple = (16,)  # 锦标赛场长：二测四个阶段都是 Rounds=16
    rounds_choices_prob: float = 1.0  # 全部按线上场长；<1 时其余在 rounds_min..rounds_max 均匀抽
    rounds_min: int = 4
    rounds_max: int = 20
    you_cai_bi_kao_prob: float = 0.0
    stage_probs: tuple = (0.7, 0.15, 0.15)  # qualifier, advancement, final
    lookahead: bool = True
    # ---- actors ----
    train_workers: int = 18
    eval_workers: int = 2
    tables_per_worker: int = 32
    samples_per_worker: int = 8192  # closed transitions per worker per iteration
    batch_wait_ms: float = 12.0  # how long the server waits to accumulate workers
    batch_target_fraction: float = 0.67  # stop waiting once this share of workers is ready
    worker_nice: int = 0
    eval_rounds: int = 16
    # ---- league ----
    self_seat_probs: tuple = (0.25, 0.25, 0.25, 0.25)  # P(k current-policy seats), k=1..4
    fixed_opponent_prob_start: float = 0.3
    fixed_opponent_prob_end: float = 0.1
    fixed_opponent_anneal_hours: float = 6.0
    league_capacity: int = 20
    league_active_slots: int = 3  # checkpoints sampled into play per iteration
    league_snapshot_minutes: float = 30.0
    pfsp_power: float = 2.0
    opponent_sampling: str = "sample"  # sample | greedy for checkpoint opponents
    include_auto_research: bool = True
    auto_research_project: str = "projects/agent-research"
    # ---- reward ----
    shaping_weight: float = 0.5
    shaping_anneal_fraction: float = 0.2  # of total planned hours
    gamma: float = 1.0
    gae_lambda: float = 0.95
    # ---- ppo ----
    ppo_epochs: int = 3
    minibatch: int = 4096
    learning_rate: float = 3e-4
    lr_final_fraction: float = 0.1
    clip: float = 0.2
    value_coef: float = 0.5
    aux_coef: float = 0.1
    entropy_coef: float = 0.01
    entropy_final: float = 0.001
    max_grad_norm: float = 1.0
    # ---- value head ----
    value_bins: int = 41
    value_range: float = 10.0  # symlog space
    # ---- model sizes ----
    mlp_hidden: int = 768
    mlp_layers: int = 3
    resnet_channels: int = 128
    resnet_blocks: int = 10
    tf_dim: int = 128
    tf_layers: int = 4
    tf_heads: int = 4
    tf_events: int = 24  # most recent event tokens fed to the transformer
    event_embed: int = 16
    autocast: bool | None = None  # bf16 mixed precision; default: all but mlp
    # ---- schedule / io ----
    hours: float = 1.0
    checkpoint_minutes: float = 10.0
    log_every: int = 1
    device: str = "cuda"
    threads: int = 8
    extra: dict = field(default_factory=dict)

    @property
    def use_autocast(self) -> bool:
        return self.autocast if self.autocast is not None else self.arch != "mlp"

    @property
    def run_dir(self) -> Path:
        return PROJECT / "runs" / self.run_name

    def as_dict(self):
        return asdict(self)


def load(path=None, **overrides) -> Config:
    data = {}
    if path:
        with open(path, "rb") as stream:
            data = tomllib.load(stream)
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f for f in Config.__dataclass_fields__}
    extra = {k: v for k, v in data.items() if k not in known}
    kwargs = {k: v for k, v in data.items() if k in known}
    for key in ("stage_probs", "self_seat_probs", "rounds_choices"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    cfg = Config(**kwargs)
    cfg.extra = extra
    return cfg
