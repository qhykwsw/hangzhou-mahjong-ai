# 策略项目

`projects/` 保存迭代线与 RL 训练框架的源码和配置。共享麻将引擎、规则、SDK、执行器与平台适配在根目录 `src/mahjong_ai/`。

## 目录约定

```text
projects/
├── agent-research/       # 不含源码：.local/research/ 下是模型库与运行时库（已退役自动研究留下的路径，ADR-0019）
└── personal/
    ├── firstprinciples/  # 迭代 Agent 推进的规则线与 RL 线（ADR-0016）
    └── selfplay/         # RL 线的训练框架：规则教师指导（guided/）与训练方案接口（recipes/API.md）
```

仓库只保留源码和配置，模型、运行数据和记录在本机生成，不进入 Git。
