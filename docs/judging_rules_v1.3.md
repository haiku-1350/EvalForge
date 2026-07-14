# EvalForge v1.3 评判规则

## 1. 职责边界

- LLM 负责语义等价、事实正确性、完整性、冲突和额外陈述判断。
- Python 负责格式、字段、ID、原文证据、确定性计算和逻辑一致性。
- 同义词、改写、语序和简称不得因为词面不同而扣分。
- `answer_type` 和人工期望只用于评分后的离线验收，不得进入模型提示词。

## 2. 参考答案原子化

每个问题的参考答案只原子化一次并缓存：

```json
{
  "question_id": "Q001",
  "reference_defect": false,
  "reference_defect_reasons": [],
  "key_points": [
    {"id": "K1", "statement": "列表创建后可以修改元素", "importance": "core"},
    {"id": "K2", "statement": "列表通常使用方括号表示", "importance": "supporting"}
  ]
}
```

每个知识点只能包含一个事实，不能重复或超出参考答案。至少需要一个 `core`。参考答案存在歧义、矛盾、事实缺陷或不足以评分时，设置 `reference_defect=true` 并转人工。

## 3. 逐点语义状态

- `matched`：完整表达，允许同义改写；必须引用原文。
- `partial`：有所表达但不完整或轻微不精确；必须引用原文。
- `missing`：没有涉及；evidence 必须为 `null`。
- `contradicted`：明确相反或冲突；必须引用冲突原文。

额外陈述使用 `correct_relevant`、`irrelevant`、`incorrect`、`unsupported`。错误陈述标记 `minor` 或 `major`，其他陈述 severity 必须为 `none`。

## 4. 模型输出协议

```json
{
  "question_id": "Q001",
  "key_point_judgments": [
    {
      "id": "K1",
      "status": "matched",
      "evidence": "列表可以修改",
      "explanation": "与列表可变语义等价"
    }
  ],
  "extra_claims": [],
  "candidate_self_contradiction": false,
  "score": 4,
  "reason": "核心事实正确，仅遗漏补充信息。",
  "uncertain": false,
  "uncertainty_reasons": []
}
```

模型 A 和 B 使用相同协议。

## 5. 分数量表

- 5：所有核心和补充知识点完整正确，没有冲突或实质错误。
- 4：所有核心点正确，仅补充信息缺失或轻微不精确。
- 3：至少一个核心点正确，但缺少重要信息。
- 2：仅有少量正确信息，主要不完整或错误。
- 1：主要结论错误，只有零散正确信息。
- 0：完全错误、无效、答非所问或拒答。

## 6. Python 硬校验

错误码包括：

- `invalid_json`
- `schema_invalid`
- `score_out_of_range`
- `missing_key_point_id`
- `duplicate_key_point_id`
- `unexpected_key_point_id`
- `evidence_not_found`
- `score_state_conflict`
- `self_contradiction_conflict`
- `judge_uncertain`
- `judge_disagreement`
- `reference_defect`

硬冲突：

- 5 分但存在 `partial`、`missing` 或 `contradicted`。
- 4 分但核心点为 `missing` 或 `contradicted`。
- 4～5 分但存在 major 级错误陈述。
- 0 分但存在明确 `matched` 的核心点。

Python 不使用简单覆盖阈值重算 3 分及以下结果。

## 7. 辅助指标

语义覆盖率：

```text
semantic_coverage = Σ(知识点权重 × 状态系数) / Σ(知识点权重)
```

- `core` 权重 2，`supporting` 权重 1。
- `matched` 系数 1，`partial` 系数 0.5，`missing` 和 `contradicted` 系数 0。

`lexical_overlap` 是规范化文本相似度，只用于调试和回归分析。高语义覆盖、低词面重合可产生 `lexical_semantic_mismatch` warning，但不能修改分数或触发人工复核。

## 8. 双模型选择

1. 模型 A 初评通过硬校验且不确定性为 false：采用 A。
2. A 失败或不确定：只把明确错误码和原因反馈给 A，纠正一次。
3. A 仍失败或不确定：模型 B 独立盲审。
4. B 通过且不确定性为 false：采用 B。
5. B 仍失败、明确不确定、与有效 A 结果相差至少 2 分，或核心点出现 `matched`/`contradicted` 直接冲突：转人工。

模型必须将“待评答案自身包含不能同时成立的陈述”与“待评答案不符合参考答案”分开判断。前者设置 `candidate_self_contradiction=true` 和 `uncertain=true`；Python 检查两者一致性并确保其进入复核链路。

模型 B 不得看到模型 A 的分数、理由、知识点状态或离线测试标签。

## 9. 审计记录

每次调用保存：模型、阶段、耗时、Token 用量、原始输出、校验失败原因和 warning。最终结果保存评分、理由、逐点判断、额外陈述、语义覆盖率、词面重合度、来源模型、状态和人工复核原因。

## 10. 验收案例

- Q001 的部分答案只表达“列表可变、元组不可变”时，核心点完整、补充表示方式缺失，应自动接受 4 分。
- Q002 的正确改写完整表达 404 与 500 含义时，应自动接受 5 分。
- 低词面重合不得单独触发 `needs_review`。
- 明确错误答案应稳定获得低分。
- 内部自相矛盾或无法确定最终立场的答案应进入人工复核。
