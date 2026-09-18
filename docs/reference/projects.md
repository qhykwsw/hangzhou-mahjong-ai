# 策略项目与共享比赛

策略源码按算法方向放在 `projects/`，共享引擎和平台能力仍在根目录 `src/mahjong_ai/`。每个叶子项目通过 `project.toml` 声明身份、算法族、入口和依赖；训练过程可以使用项目自己的 `models/` 与 `runs/`，发布结果写入不可变的 `releases/`。

自动研究项目是 `projects/agent-research/`。它继续使用 `research.toml` 管理组合、分支和研究轮次。人工维护的自我对弈、规则或搜索项目不进入自动研究状态机，但可以构建冻结发布版本。

`CompetitionRegistry` 使用独立的 `.local/competition/competition.sqlite` 保存发布版本注册和常驻对手状态；独立比赛的详细结果保存为 `.local/competition/runs/*.json`。注册表不修改 `research.sqlite`。版本注册后不能覆盖；暂停或撤销只影响未来比赛。人工版本加入自动研究后，标记为常驻对手，直到明确移除。

构建和注册人工版本：

```bash
.venv/bin/mahjong-ai release-build \
  --project projects/personal/rules --version v0.1.0
.venv/bin/mahjong-ai competition-register \
  projects/personal/rules/releases/v0.1.0
.venv/bin/mahjong-ai competition-list
```

只有显式追加 `--resident`，冻结版本才会在后续自动研究轮次中作为常驻对手；省略该选项时，它不会进入常驻对手池。

独立大乱斗传入四个参赛策略，可组合冻结发布版本与内置基线；输出的 `standings` 按场均桌面分排序，同分看场均名次分。接着使用上面构建的规则版本：

```bash
.venv/bin/mahjong-ai competition-run \
  --policies projects/personal/rules/releases/v0.1.0 rule rule rule \
  --games 16 --output .local/competition/runs/brawl-01.json
```

仓库只提供项目清单与源码，不附带训练模型、实验数据或冻结发布包。自我对弈项目先用 `projects/personal/selfplay/train/run.py` 训练并导出 ONNX，把导出复制到该项目的 `models/` 后再构建发布包，训练器、评估与目录约定见 [项目说明](../../projects/personal/selfplay/README.md) 与 [ADR-0009](../adr/0009-independent-selfplay-trainer-for-personal-project.md)；规则项目可以直接构建。各机器重新建立依赖环境与发布版本，`models/`、`releases/`、`runs/` 和 `.local/` 等运行产物不提交 Git，须独立保存。项目必须从项目清单创建，不支持无清单的旧布局。
