from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0
MAX_TOKENS = 500
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """你是严格、稳定的答案质量评测器。请比较待评估答案与参考答案，只评估事实和语义，不因措辞不同扣分。

评分标准：
5：与参考答案语义一致，关键内容完整，没有错误。
4：核心结论正确，只有不影响使用的轻微遗漏。
3：部分正确，但缺少重要信息。
2：包含少量正确信息，但主要内容不完整或存在明显错误。
1：基本错误，只与问题有少量关联。
0：完全错误、答非所问、拒绝回答或与参考答案矛盾。

只返回一个 JSON 对象，不要使用 Markdown，不要添加 JSON 之外的文字。格式必须严格为：
{"score": 0到5的整数, "reason": "具体评分理由"}

reason 必须明确指出待评估答案与参考答案相比，哪些关键内容正确、遗漏或错误；不得为空，不得只复述分数。相同输入应采用相同尺度。"""


class EvaluationError(RuntimeError):
    """Raised when the API response cannot produce a valid evaluation."""


@dataclass(frozen=True)
class EvaluationResult:
    score: int
    reason: str


HttpPost = Callable[[str, dict[str, str], bytes, int], dict[str, Any]]


def _default_http_post(
    url: str, headers: dict[str, str], body: bytes, timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EvaluationError(f"DeepSeek API 返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise EvaluationError(f"无法连接 DeepSeek API：{exc.reason}") from exc
    except TimeoutError as exc:
        raise EvaluationError("调用 DeepSeek API 超时") from exc

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise EvaluationError("DeepSeek API 返回了无法解析的响应") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError("DeepSeek API 响应不是 JSON 对象")
    return decoded


class DeepSeekEvaluator:
    def __init__(self, http_post: HttpPost | None = None) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EvaluationError(
                "未检测到环境变量 DEEPSEEK_API_KEY。请先配置该环境变量后再运行。"
            )
        self._api_key = api_key
        self._http_post = http_post or _default_http_post

    def evaluate(
        self, question: str, reference_answer: str, candidate_answer: str
    ) -> EvaluationResult:
        user_prompt = (
            "请按既定标准评测以下答案，并返回 JSON。\n"
            f"问题：{question}\n"
            f"参考答案：{reference_answer}\n"
            f"待评估答案：{candidate_answer}"
        )
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: EvaluationError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._http_post(
                    API_URL, headers, body, REQUEST_TIMEOUT_SECONDS
                )
                return self._parse_api_response(response)
            except EvaluationError as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS:
                    time.sleep(0.5 * attempt)
        raise EvaluationError(
            f"连续 {MAX_ATTEMPTS} 次未获得合法评测结果：{last_error}"
        ) from last_error

    @staticmethod
    def _parse_api_response(response: dict[str, Any]) -> EvaluationResult:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvaluationError("API 响应缺少 choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise EvaluationError("评测模型返回了空内容")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise EvaluationError("评测模型返回的内容不是合法 JSON") from exc
        return validate_result(result)


def validate_result(result: Any) -> EvaluationResult:
    if not isinstance(result, dict):
        raise EvaluationError("评测结果必须是 JSON 对象")
    if set(result) != {"score", "reason"}:
        raise EvaluationError("评测结果必须且只能包含 score 和 reason")

    score = result["score"]
    reason = result["reason"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
        raise EvaluationError("score 必须是 0 到 5 之间的整数")
    if not isinstance(reason, str) or not reason.strip():
        raise EvaluationError("reason 必须是非空字符串")
    return EvaluationResult(score=score, reason=reason.strip())
