# EvalForge v2.1

EvalForge v2.1 通过 Python 接口调用真实 RAG 系统，同时评测最终答案正确性、检索忠实度并确定错误来源。测试数据只保存三个问题及基于 RAG 数据库的参考答案，不预置待评答案。

当前默认接入：

- RAG 项目：`E:\Enterprise AI Helpdesk`
- Python 入口：`utils.answer:answer_user`
- GitHub 项目：`haiku-1350/AI-Helpdesk-Assistant`

## v2.1 流程

```text
问题 + 数据库参考答案
    ↓ Python 接口并捕获轨迹
Router → 改写问题 → 检索内容 → 最终回答
    ↓
correctness：最终回答 vs 参考答案
retrieval：检索内容 vs 参考答案
groundedness：最终回答中的陈述 vs 检索内容
    ↓ Python 确定性归因
error_type：retrieval / generation / both / none
```

三个 LLM 判断均使用模型 A、Python 硬校验、A 纠正一次、模型 B 独立盲审和必要时人工复核的流程。

## 错误归因

默认通过门槛为 4 分：

| 检索是否通过 | 回答是否有检索支持 | 归因 |
| --- | --- | --- |
| 否 | 是 | `retrieval` |
| 是 | 否，或最终答案不正确 | `generation` |
| 否 | 否 | `both` |
| 是 | 是，且最终答案正确 | `none` |

`retrieval_score` 是归因所需的诊断字段。用户要求的正式输出字段为：

```json
{
  "correctness_score": 5,
  "groundedness_score": 5,
  "error_type": "none"
}
```

## 三个测试问题

默认数据位于 `data/test_cases.json`：

1. 员工请假申请流程
2. VPN 无法连接的处理方式
3. 报销所需材料

每条数据严格包含 `question_id`、`question` 和 `reference_answer`。`answers` 等离线待评答案字段会被拒绝。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

环境变量：

- `DEEPSEEK_API_KEY`：RAG Router、RAG 回答生成和 EvalForge 模型 A 共用
- `GLM_API_KEY`：模型 B，在 A 纠正仍未通过时使用

## 运行

```powershell
.\.venv\Scripts\python.exe main.py
```

指定其他本地 RAG 项目或入口：

```powershell
.\.venv\Scripts\python.exe main.py `
  --rag-project "E:\other-rag" `
  --rag-entrypoint "package.api:answer" `
  --min-score 4
```

也可以使用 `EVALFORGE_RAG_PROJECT` 和 `EVALFORGE_RAG_ENTRYPOINT`。

v2.1 必须取得检索轨迹。入口可以像当前 Demo 一样在模块全局暴露 `route_query` 和 `retrieve`，适配器会在不修改 RAG 源码的前提下捕获调用；也可以直接返回包含 `answer`、`intent`、`rewritten_question`、`retrieved_context` 和 `need_human` 的字典。

## 输出和退出码

默认结果保存在 `results/rag_evaluation_results.json`，包括 Router 意图、改写问题、检索内容、最终回答、三个分数、错误归因和全部模型审计记录。

- `0`：三个结果均为 `error_type=none` 且无需人工复核
- `1`：存在 RAG 错误或人工复核项
- `2`：RAG 接口、配置、API 或文件错误
- `130`：用户取消运行

评分细则见 `docs/judging_rules_v1.3.md`，RAG 接入约定见 `docs/rag_integration_v2.md`，v2.1 归因规则见 `docs/error_attribution_v2.1.md`。
