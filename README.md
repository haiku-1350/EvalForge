# EvalForge v1.3

EvalForge v1.3 使用“参考答案原子化 + LLM 语义判断 + Python 硬校验 + 选择性双模型盲审”评测短文本问答。

## v1.3 变化

- 参考答案按问题拆成 `core` 和 `supporting` 原子知识点，并在同一轮运行中缓存复用。
- 模型必须逐个知识点输出 `matched`、`partial`、`missing` 或 `contradicted`。
- 除 `missing` 外，每个状态必须引用待评答案中的真实原文证据。
- Python 只验证 JSON、字段、证据、知识点 ID、分数状态冲突和不确定性。
- 待评答案自身存在矛盾时，模型单独输出 `candidate_self_contradiction=true`；Python 强制要求同时设置 `uncertain=true` 并进入复核链路。
- 原“关键词覆盖率”改为 `lexical_overlap`，只用于诊断，不修改分数、不单独触发人工复核。
- 新增按知识点状态加权计算的 `semantic_coverage`，同样不直接生成分数。
- 每次模型调用保存模型、阶段、耗时、Token、原始输出、失败原因和 warning。
- 默认保存结构化结果到 `results/evaluation_results.json`。

完整实现规则见 [docs/judging_rules_v1.3.md](docs/judging_rules_v1.3.md)。

## 双模型流程

```text
模型 A 首次评分
    ↓ Python 硬校验失败或模型不确定
模型 A 根据明确违规原因纠正一次
    ↓ 仍失败或不确定
模型 B 独立盲审
    ↓ 仍失败、明确不确定或与 A 重大分歧
人工复核
```

模型 B 只看到问题、参考答案、缓存的原子知识点和待评答案，不会看到模型 A 的分数、理由或状态。

## 环境变量

- `DEEPSEEK_API_KEY`：模型 A 密钥
- `GLM_API_KEY`：模型 B 密钥

默认模型 A 为 `deepseek-v4-flash`，模型 B 为 `glm-5.1`，模型 B 默认接口为 `https://open.bigmodel.cn/api/paas/v4/chat/completions`。

可选覆盖项：

- `EVALFORGE_MODEL_A`
- `EVALFORGE_MODEL_B`
- `EVALFORGE_MODEL_B_API_URL`
- `EVALFORGE_MODEL_B_API_KEY`

## 运行

```powershell
python main.py
python main.py --acceptance
python main.py --data data/test_cases.json --output results/custom.json
python -m unittest discover -s tests -v
```

## Python 硬校验

- 输出必须是严格 JSON，字段和类型正确。
- 分数必须是 0～5 的整数。
- 知识点 ID 必须完整对应、无重复、无新增。
- `matched`、`partial`、`contradicted` 的 evidence 必须是待评答案中的原文。
- `missing` 的 evidence 必须为 `null`。
- 5 分不能存在非 `matched` 知识点。
- 4 分不能存在 `missing` 或 `contradicted` 的核心点。
- 4～5 分不能存在 major 级错误陈述。
- `uncertain=true` 必须提供原因，并进入后续复核。
- `candidate_self_contradiction=true` 时 `uncertain` 不能为 false。

## 退出码

- `0`：所有基础答案均得到可接受的自动结果
- `1`：至少一条结果需要人工复核，或离线验收失败
- `2`：配置、API、文件或其他系统错误
- `130`：用户取消运行

`answer_type` 只在评分完成后用于离线验收，不会发送给任何评分模型。
