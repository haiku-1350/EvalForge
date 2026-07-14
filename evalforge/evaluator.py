from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .reviewer import ReviewReport, review_content, review_result


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0
MAX_TOKENS = 500
REQUEST_TIMEOUT_SECONDS = 60
MAX_LLM_RETRIES = 2
MAX_ATTEMPTS = 1 + MAX_LLM_RETRIES

SYSTEM_PROMPT = """你是严格、稳定的答案质量评测器。请比较待评估答案与参考答案，只评估事实和语义，不因措辞不同扣分。

评分标准：
5：与参考答案语义一致，关键内容完整，没有错误。
4：核心结论正确，只有不影响使用的轻微遗漏。
3：部分正确，但缺少重要信息。
2：包含少量正确信息，但主要内容不完整或存在明显错误。
1：基本错误，只与问题有少量关联。
0：完全错误、答非所问、拒绝回答或与参考答案矛盾。

从参考答案中提取 1 到 8 个用于评分的原子关键词或短语。关键词必须原样出现在参考答案中，尽量短，并优先选择待评估答案可直接覆盖的核心概念。

只返回一个 JSON 对象，不要使用 Markdown，不要添加 JSON 之外的文字。格式必须严格为：
{"score": 0到5的整数, "reason": "具体评分理由", "keywords": ["参考答案关键词"]}

reason 必须明确指出待评估答案与参考答案相比，哪些关键内容正确、遗漏或错误；不得为空，不得只复述分数。相同输入应采用相同尺度。"""


class EvaluationError(RuntimeError):
    """Raised when the API response cannot produce a valid evaluation."""


@dataclass(frozen=True)
class EvaluationResult:
    score: int | None
    reason: str
    keywords: tuple[str, ...] = ()
    covered_keywords: tuple[str, ...] = ()
    coverage_ratio: float = 0.0
    needs_review: bool = False
    review_issues: tuple[str, ...] = ()
    attempts: int = 1


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
        base_prompt = (
            "请按既定标准评测以下答案，并返回 JSON。\n"
            f"问题：{question}\n"
            f"参考答案：{reference_answer}\n"
            f"待评估答案：{candidate_answer}"
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: EvaluationError | None = None
        last_report: ReviewReport | None = None
        retry_feedback = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_prompt + retry_feedback},
                ],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                response = self._http_post(
                    API_URL, headers, body, REQUEST_TIMEOUT_SECONDS
                )
                content = self._extract_api_content(response)
            except EvaluationError as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS:
                    time.sleep(0.5 * attempt)
                    continue
                raise EvaluationError(
                    f"连续 {MAX_ATTEMPTS} 次调用失败：{last_error}"
                ) from last_error

            report = review_content(content, reference_answer, candidate_answer)
            last_report = report
            if report.passed:
                return self._build_result(report, attempt, needs_review=False)
            if attempt < MAX_ATTEMPTS:
                retry_feedback = (
                    "\n\n上一次输出未通过 Python 复核："
                    + "；".join(report.issues)
                    + "。请重新评估并修正输出。"
                )

        if last_report is None:
            raise EvaluationError("未获得可复核的 LLM 输出")
        return self._build_result(last_report, MAX_ATTEMPTS, needs_review=True)

    @staticmethod
    def _extract_api_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvaluationError("API 响应缺少 choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise EvaluationError("评测模型返回了空内容")
        return content

    @staticmethod
    def _build_result(
        report: ReviewReport, attempts: int, needs_review: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            score=report.score,
            reason=report.reason or "LLM 输出未通过 Python 复核，需人工处理。",
            keywords=report.keywords,
            covered_keywords=report.covered_keywords,
            coverage_ratio=report.coverage_ratio,
            needs_review=needs_review,
            review_issues=report.issues,
            attempts=attempts,
        )


def validate_result(result: Any) -> EvaluationResult:
    """Validate the LLM JSON schema without answer-specific coverage checks."""
    report = review_result(result, "", "")
    non_coverage_issues = tuple(
        issue
        for issue in report.issues
        if not issue.startswith(("关键词错误：", "评分冲突："))
    )
    if non_coverage_issues:
        raise EvaluationError("；".join(non_coverage_issues))
    return EvaluationResult(
        score=report.score,
        reason=report.reason,
        keywords=report.keywords,
    )
