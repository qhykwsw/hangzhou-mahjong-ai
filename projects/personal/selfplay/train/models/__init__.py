from .common import PolicyNet, TwoHot, symexp, symlog  # noqa: F401


def build_model(cfg):
    if cfg.arch == "mlp":
        from .mlp import MLPNet as cls
    elif cfg.arch == "resnet":
        from .resnet import ResNet as cls
    elif cfg.arch == "transformer":
        from .transformer import TransformerNet as cls
    else:
        raise ValueError(f"unknown arch {cfg.arch}")
    return cls(cfg)
