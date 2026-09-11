# 策略项目与共享比赛

策略源码按算法方向放在 `projects/`，共享引擎和平台能力仍在根目录 `src/mahjong_ai/`。每个叶子项目通过 `project.toml` 声明身份、算法族、入口和依赖；训练过程可以使用项目自己的 `models/` 与 `runs/`，发布结果写入不可变的 `releases/`。

自动研究项目是 `projects/agent-research/`。它继续使用 `research.toml` 管理组合、分支和研究轮次。人工维护的 PPO、规则或搜索项目不进入自动研究状态机，但可以构建冻结发布版本。

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

独立大乱斗直接传入四个冻结版本：

```bash
.venv/bin/mahjong-ai competition-run \
  --policies projects/personal/rules/releases/v1 \
             projects/personal/ppo/releases/v1 \
             rule rule \
  --games 16 --output .local/competition/runs/brawl-01.json
```

现有自动研究迁移必须使用受控迁移工具，保持实验、分支、候选、回执和冻结 runtime 的 ID 与内容哈希。迁移前后都要验证保存的路径、包树哈希和恢复能力。
