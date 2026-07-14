# 外部 RAG HTTP 接口

EvalForge v2.2 对每个问题发送一次 HTTP POST。接口使用 UTF-8 JSON，不依赖 OpenAI 兼容格式。

## 请求

```http
POST /evalforge/query
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
```

```json
{
  "question_id": "Q001",
  "question": "员工请假需要怎么申请？"
}
```

`Authorization` 是可选项。EvalForge 只从 `EVALFORGE_RAG_API_KEY` 或 `--rag-api-key-env` 指定的环境变量读取 Token。

## 响应

必填字段：

- `answer`：非空字符串，外部 RAG 的最终回答
- `retrieved_context`：字符串，本次生成实际使用的检索内容；没有召回时返回空字符串

可选字段：

- `intent`：字符串或 `null`
- `rewritten_question`：字符串或 `null`
- `need_human`：布尔值或 `null`

示例：

```json
{
  "answer": "请提前三天在系统提交申请，主管批准后生效。",
  "retrieved_context": "请假流程：提前3天在系统提交申请，主管审批后生效",
  "intent": "HR",
  "rewritten_question": "员工请假申请流程",
  "need_human": false
}
```

响应不接受未声明字段。HTTP 错误、超时、非法 JSON、字段类型不符和超过 2 MiB 的响应都按系统错误处理，进程返回 2。

## 运行

```powershell
.\.venv\Scripts\python.exe main.py `
  --rag-transport http `
  --rag-url "https://rag.example.com/evalforge/query"
```

如果外部接口需要其他鉴权方式，应在网关层转换为 Bearer Token；EvalForge v2.2 不接收命令行明文密钥，也不提供关闭 TLS 校验的选项。
