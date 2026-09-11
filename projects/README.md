# 策略项目

`projects/` 保存策略项目的源码、训练配置和发布工作区。共享麻将引擎、规则、SDK、评估、比赛与平台适配仍在根目录 `src/mahjong_ai/`。

## 目录约定

```text
projects/
├── agent-research/       # 自动研究项目；组合仍由 research.toml 注册
└── personal/
    ├── ppo/              # 人工维护的 PPO 项目
    └── rules/            # 人工维护的规则项目
```

每个叶子项目使用 `project.toml` 声明 `id`、`family`、入口和精确依赖。项目可以按需拥有 `src/`、`models/`、`releases/` 和 `runs/`。比赛只消费 `releases/` 中的冻结发布包，不直接运行可变工作区。

自动研究项目的研究状态、实验和分支历史也归属于该项目；人工项目不创建自动研究分支。将人工版本加入自动研究时，只向共享比赛注册表登记具体冻结发布版本，并按用户要求标记为常驻对手。
