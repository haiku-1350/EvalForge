# EvalForge v1.1

这是一个带 Python 复核的 LLM 答案评测流程：从 JSON 读取问题、参考答案和三类 RAG 待评估答案，调用固定的 DeepSeek 模型评分并提取参考答案关键词，再用确定性的 Python 规则复核结果。

## 复核流程

LLM 必须返回以下结构：

```json
{"score": 4, "reason": "核心结论正确，但有轻微遗漏。", "keywords": ["关键词一", "关键词二"]}
```

Python 会依次检查：

- JSON 格式以及字段是否严格为 `score`、`reason`、`keywords`
- `score` 是否为 0 到 5 的整数
- 关键词是否为 1 到 8 个不重复的非空字符串，且原样来自参考答案
- RAG 待评答案的关键词覆盖率是否足以支持 LLM 分数

覆盖率最低要求为：2 分至少覆盖 1 个关键词，3 分至少 50%，4 分至少 75%，5 分必须 100%。0 到 1 分不做反向覆盖限制，因为包含关键词的答案仍可能通过否定、错配等方式与参考答案矛盾。

首次结果未通过复核时，程序会把具体问题反馈给 LLM，最多重新调用 2 次。第三次结果仍违规时返回 `needs_review=True`，保留可解析的分数、关键词、覆盖率和违规原因，交给人工复核。API 连接或响应信封连续失败时仍抛出运行错误。

## 运行条件

- Python 3.10 或更高版本
- 环境变量 `DEEPSEEK_API_KEY` 已配置
- 不需要安装第三方 Python 包

程序只通过 `os.getenv("DEEPSEEK_API_KEY")` 读取密钥，不读取 `.env` 或其他本地密钥文件，也不会打印密钥。

## 运行

```powershell
python main.py
python main.py --acceptance
python -m unittest discover -s tests -v
```

默认数据位于 `data/test_cases.json`。也可以用 `--data <路径>` 指定同结构的 JSON 文件。

## 固定评测配置

- 模型：`deepseek-v4-flash`
- 温度：`0`
- Thinking：关闭
- 最大输出：`500` tokens
- 输出模式：`json_object`
- 评分提示词：固定在 `evalforge/evaluator.py` 中
- Python 复核规则：固定在 `evalforge/reviewer.py` 中
- LLM 纠正调用：最多 `2` 次（包括首次调用最多 `3` 次）

当前版本只在终端展示结果，不实现 RAG 检索本身、报告、双模型或界面。
