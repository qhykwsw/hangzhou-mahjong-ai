# RL 训练框架（selfplay）

RL 线的训练框架，用 PyTorch 训练：

- **规则教师指导**（[`guided/`](guided/)）：以冻结的规则版本为教师，先纯模仿，再用真实桌面分做强化学习并保留按师生差距调整的教师辅助损失；Actor 由独立的推理服务出牌。用法见[训练手册](guided/README.md)，决定见 [ADR-0011](../../../docs/adr/0011-rule-guided-policy-learning.md)。
- **训练方案接口**（[`recipes/API.md`](recipes/API.md)）：RL 线每次尝试修改的训练方案（`recipe` 包）在 `guided/trial.py` 里运行时能用到的上下文与约束。RL 线自己的方案在 `../firstprinciples/rl/recipe/`。

早期的自我对弈 PPO 训练器（`train/`，ONNX 发布）和自动研究里的方案分支已删除，见 [ADR-0019](../../../docs/adr/0019-retire-automated-research.md)。术语（桌面分、阶段总分、教师策略）见 [CONTEXT.md](../../../CONTEXT.md)。

## 目录

```text
selfplay/
├── src/selfplay_core/  # actions（231 动作编号）、encoding（观察编码）、wire（推理服务的线路格式）；只依赖 numpy 与 mahjong_ai
├── guided/             # 训练框架，依赖 torch；见 guided/README.md
├── recipes/API.md      # 训练方案接口
├── tests/              # 项目内测试
└── runs/               # 训练产物，不提交
```

## 约定

- `src/` 里不能出现 torch；训练与推理服务都从 `src/selfplay_core` 导入编码，保证两边一致。
- 训练进程里价值网络与信念头的训练标签可以读隐藏状态（[ADR-0009](../../../docs/adr/0009-independent-selfplay-trainer-for-personal-project.md)），oracle guiding 的读取强度在训练中衰减到零、导出时剥离（[ADR-0018](../../../docs/adr/0018-oracle-guided-policy-training.md)）；Actor 与所有推理只读合法观察。
- 改 `guided/` 后按[迭代线说明](../firstprinciples/README.md)给每台 RL 主机重新 `setup` 并封存。

```bash
# 项目测试
.venv/bin/python -m pytest -q projects/personal/selfplay/tests
```
