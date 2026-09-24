# 策略项目

`projects/` 保存策略项目的源码、训练配置和发布工作区。共享麻将引擎、规则、SDK、评估、比赛与平台适配仍在根目录 `src/mahjong_ai/`。

## 目录约定

```text
projects/
├── agent-research/       # 自动研究项目；组合仍由 research.toml 注册
└── personal/
    ├── selfplay/         # 人工维护的策略训练项目：自我对弈 PPO（ONNX 发布）与规则教师指导（guided/）
    ├── firstprinciples/  # 维护者亲自推进的规则线与 RL 线（ADR-0016）
    └── rules/            # 人工维护的规则项目
```

仓库只保留项目清单、配置和源码，运行数据在本机生成。清单约定、训练、冻结发布、比赛注册与常驻对手操作统一见[策略项目与共享比赛](../docs/reference/projects.md)；自动研究操作见[研究手册](../docs/reference/strategy-agent.md)。
