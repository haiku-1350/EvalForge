# EvalForge v1.2

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

评测链路固定为：

1. 模型 A 首次评分，Python 复核。
2. 失败时把 Python 违规原因交给模型 A，允许纠正一次。
3. 仍失败时调用模型 B 独立盲审。模型 B 只接收原问题、参考答案和待评答案，不接收模型 A 的输出或违规原因。
4. 模型 B 仍未通过 Python 复核时返回 `needs_review=True`，交给人工复核。

默认模型 A 为 `deepseek-v4-flash`，模型 B 为智谱 `glm-5.1`。模型 B 使用智谱通用 Chat Completions 接口 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，并且只在模型 A 纠正失败后才读取和检查凭据。

- `GLM_API_KEY`：模型 B 的默认 API 密钥
- `EVALFORGE_MODEL_B`：可选，覆盖模型 B 名称
- `EVALFORGE_MODEL_B_API_URL`：可选，覆盖完整的 Chat Completions 接口地址
- `EVALFORGE_MODEL_B_API_KEY`：可选，优先于 `GLM_API_KEY` 覆盖模型 B 密钥

模型 B 未配置不会影响模型 A 的正常评分；只有流程实际进入盲审阶段时才会报出缺少的配置。API 连接或响应信封失败属于技术错误，不会被当成评分冲突转交下一模型，也不会直接设置 `needs_review`。

## 运行条件

- Python 3.10 或更高版本
- 环境变量 `DEEPSEEK_API_KEY` 已配置
- 不需要安装第三方 Python 包

程序只通过 `os.getenv("DEEPSEEK_API_KEY")` 读取密钥，不读取 `.env` 或其他本地密钥文件，也不会打印密钥。

## 运行

评测全部 9 条测试答案：

```powershell
python main.py
```

执行完整验收，包括三条固定样本各运行 3 次的稳定性检查：

```powershell
python main.py --acceptance
```

运行不调用 API 的本地单元测试：

```powershell
python -m unittest discover -s tests -v
```

默认数据位于 `data/test_cases.json`。也可以用 `--data <路径>` 指定同结构的 JSON 文件。

## 固定评测配置

- 模型 A：`deepseek-v4-flash`
- 模型 B：`glm-5.1`，使用独立的 `GLM_API_KEY`
- 温度：`0`
- Thinking：关闭
- 最大输出：`500` tokens
- 输出模式：`json_object`
- 评分提示词：固定在 `evalforge/evaluator.py` 中
- Python 复核规则：固定在 `evalforge/reviewer.py` 中
- 模型 A 纠正：最多 `1` 次
- 模型 B 盲审：模型 A 纠正失败后调用 `1` 次

当前版本只在终端展示结果，不实现 RAG 检索本身、报告、双模型或界面。
