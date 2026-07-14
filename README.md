# EvalForge V1

这是一个最小可运行的 LLM 答案评测流程：从 JSON 读取问题、参考答案和三类待评估答案，调用固定的 DeepSeek 模型，并严格解析模型返回的 JSON。

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

执行完整 V1 验收，包括三条固定样本各运行 3 次的稳定性检查：

```powershell
python main.py --acceptance
```

运行不调用 API 的本地单元测试：

```powershell
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

V1 只在终端展示结果，不实现 RAG、报告、错误归因、双模型或界面。
