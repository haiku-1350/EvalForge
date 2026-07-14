# EvalForge v2.1 错误归因规则

## 1. 三个判断

### correctness_score

最终回答相对于数据库参考答案的正确性，范围 0～5。沿用 v1.3 的原子知识点、原文证据和分数状态校验。

### retrieval_score

检索内容相对于数据库参考答案的覆盖质量，范围 0～5。该字段用于确定错误来源，并作为诊断数据保存。

### groundedness_score

最终回答中的事实陈述是否得到检索内容支持，范围 0～5。该判断不要求回答复述全部检索内容，只检查回答实际说出的内容。

每条回答陈述必须标记为：

- `supported`
- `partially_supported`
- `unsupported`
- `non_factual`

除 `unsupported` 和 `non_factual` 外，必须引用检索原文证据。所有陈述都必须引用最终回答原文。

## 2. error_type

门槛默认是 4 分，人工复核或无有效分数视为未通过。

- 检索未通过、groundedness 通过：`retrieval`
- 检索通过，但 correctness 或 groundedness 未通过：`generation`
- 检索和 groundedness 均未通过：`both`
- 三项均通过：`none`

`error_type` 由 Python 根据三个结果确定，不由 LLM 直接输出。

## 3. 空检索内容

检索为空时，资料性事实不能标记为有支持。最终回答如果诚实说明没有信息，可将该陈述标记为 `non_factual` 并获得较高 groundedness；如果 correctness 同时不足，则归因为 `retrieval`。

## 4. 双模型流程

groundedness 判断同样执行：模型 A 首评、Python 校验、A 纠正一次、模型 B 独立盲审、必要时人工复核。模型 B 不查看 A 的评分和理由。
