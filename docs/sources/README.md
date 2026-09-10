# 平台原始资料

这里保留平台文档、示例和接口响应，供协议排查与回归测试追溯。**所有资料只保存最新一份，使用固定文件名；更新时必须整套刷新，不能只更新指南或修改版本标签。** 当前版本号与官方变更历史见[版本信息](platform-guide-version.json)。现行用法见[平台规则与接入](../reference/platform.md)。

| 文件 | 来源与用途 |
| --- | --- |
| [最新指南](platform-guide.html) | `/portal/api/guide` 返回的 HTML 正文，原样解码保存 |
| [版本信息](platform-guide-version.json) | `/portal/api/guide/version` 的完整版本与变更信息 |
| [赛事流程](platform-flow.html) | 当前门户 `section#view-flow` 的原始 HTML |
| [最小 Bot 示例](official-minimal-bot.py) | 从最新指南提取的原始示例；不包含完整策略和并发调度 |
| [回放逻辑](platform-replay.js) | 当前门户 `/portal/replay.js`；用于赛后回放，并非完整对局引擎 |
| [四白板番型响应](fan-calc-four-whites.json) | 重新请求无状态 `/portal/api/tools/fan-calc` 的输入与响应；由规则回归测试直接引用 |
| [番型校准](fan-calibration.json) | 重新运行 32 组本地计算与当前官方无状态响应的对照，保存真实一致性结果 |

原始文本保持原貌；发现指南、版本日志与响应冲突时，核对证据后更新现行速查。无状态番型计算不能证明全部合法动作和状态转换等价。

## 整套更新

在项目根目录运行：

```bash
.venv/bin/python scripts/update_platform_sources.py --insecure
.venv/bin/python -m pytest -q tests/test_rules.py
```

[更新脚本](../../scripts/update_platform_sources.py)获取上表全部七类资料，重新提取 Bot、重新请求番型接口，并在更新前后核对指南版本一致。全部获取完成才替换文件；请求失败不会覆盖现有资料。校准不一致时仍保存真实新证据并报错，应定位差异，不能保留旧结果冒充最新。

同步检查参考手册中的规则、字段与实现边界，修正受影响的说明和测试引用。来源正文中的日期与版本历史保持原貌；重新获取不代表上游内容一定改变，也不代表本地引擎已适配新规则。新增资料必须同时纳入脚本和本表，避免后续漏更。实验成绩、性能测试和运行总结放 `.local/research/reports/`。
