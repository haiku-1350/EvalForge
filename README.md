# EvalForge v2

EvalForge v2 通过 Python 接口调用真实 RAG 系统，将 RAG 生成的回答交给 v1.3 的语义评分和双模型复核流程。测试数据只保存三个问题及参考答案，不再预置待评答案。

当前默认接入：

- RAG 项目：`E:\Enterprise AI Helpdesk`
- Python 入口：`utils.answer:answer_user`
- GitHub 项目：`haiku-1350/AI-Helpdesk-Assistant`

## 流程

```text
问题 + 参考答案
    ↓ Python 接口
RAG Router → 检索 → 回答生成
    ↓ 实时回答
参考答案原子化 → 模型 A 评分 → Python 硬校验
    ↓ 失败或不确定
模型 A 纠正一次 → 模型 B 盲审 → 人工复核
```

EvalForge 不复制或修改 RAG 项目代码。`PythonRagAdapter` 在 RAG 项目目录下调用指定函数，因此 Demo 中基于相对路径读取 `data/*.txt` 的逻辑可以正常工作。

## 三个测试问题

默认数据位于 `data/test_cases.json`：

1. 员工请假申请流程
2. VPN 无法连接的处理方式
3. 报销所需材料

每条数据严格包含：

```json
{
  "question_id": "Q001",
  "question": "员工请假需要怎么申请？",
  "reference_answer": "员工需要提前 3 天在系统提交请假申请，主管审批后生效。"
}
```

`answers` 等离线待评答案字段会被拒绝。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

环境变量：

- `DEEPSEEK_API_KEY`：RAG Router、RAG 回答生成和 EvalForge 模型 A 共用
- `GLM_API_KEY`：EvalForge 模型 B，在 A 纠正仍未通过时使用

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

也可以使用：

- `EVALFORGE_RAG_PROJECT`
- `EVALFORGE_RAG_ENTRYPOINT`

RAG 入口函数必须接收一个问题字符串并返回非空回答字符串。

## 输出和退出码

默认结果保存在 `results/rag_evaluation_results.json`，其中包括：

- 问题、参考答案和实时 RAG 回答
- RAG 项目、Python 入口和调用耗时
- 最终分数、理由、语义覆盖和词面重合
- 每次模型调用的模型、阶段、耗时、Token、原始输出及校验问题

退出码：

- `0`：三个回答均无需人工复核，且达到最低分
- `1`：存在低于最低分或需要人工复核的回答
- `2`：RAG 接口、配置、API 或文件错误
- `130`：用户取消运行

v1.3 的评分细则仍见 `docs/judging_rules_v1.3.md`，v2 接入约定见 `docs/rag_integration_v2.md`。
