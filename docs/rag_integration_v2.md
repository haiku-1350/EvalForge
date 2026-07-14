# EvalForge v2 RAG 接入约定

## 1. 接口边界

EvalForge 通过 `module:function` 形式加载本地 Python 接口。函数签名为：

```python
def answer(question: str) -> str:
    ...
```

当前 Demo 对应 `utils.answer:answer_user`。适配器验证项目目录、模块来源、函数可调用性、输入问题和返回答案，不接管 RAG 内部的 Router、检索器或生成模型。

## 2. 工作目录

调用期间工作目录切换到 RAG 项目根目录，调用结束后恢复。这样可以兼容 Demo 中的 `data/hr.txt`、`data/it.txt`、`data/finance.txt` 等相对路径。

工作目录切换由进程级锁保护。当前 CLI 串行评测三个问题，不并发调用 RAG。

## 3. 数据契约

v2 输入必须恰好包含三个问题，每个问题只能包含：

- `question_id`
- `question`
- `reference_answer`

待评答案由 RAG 运行时产生，输入中不得包含 `answers` 或 `candidate_answer`。

## 4. 评测与验收

每个实时回答沿用 v1.3 规则：参考答案原子化、模型 A 首评、Python 硬校验、A 纠正一次、模型 B 独立盲审、必要时转人工。

默认自动验收门槛为 4 分。出现以下任一情况时进程返回 1：

- 分数低于门槛；
- 评分没有有效分数；
- `needs_review=true`。

## 5. 审计数据

结果记录 RAG 实时回答、项目路径、入口、调用耗时，以及 EvalForge 原有的参考答案原子化和评分尝试明细。API 密钥不进入结果文件、日志或模型提示词。
