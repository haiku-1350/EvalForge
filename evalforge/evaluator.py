from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .reviewer import ReviewReport, review_content, review_result


API_URL = "https://api.deepseek.com/chat/completions"
MODEL_A = "deepseek-v4-flash"
MODEL_B = ""
# Backward-compatible alias for callers that only display the primary model.
MODEL = MODEL_A
TEMPERATURE = 0
MAX_TOKENS = 500
REQUEST_TIMEOUT_SECONDS = 60

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
class EvaluationAttempt:
    stage: str
    model: str
    passed: bool
    score: int | None
    issues: tuple[str, ...]


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
    model: str = MODEL_A
    stage: str = "model_a_initial"
    review_history: tuple[EvaluationAttempt, ...] = ()


HttpPost = Callable[[str, dict[str, str], bytes, int], dict[str, Any]]


def _default_http_post(
    url: str, headers: dict[str, str], body: bytes, timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Do not expose request headers or the API key in exceptions/logs.
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EvaluationError(f"模型 API 返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise EvaluationError(f"无法连接模型 API：{exc.reason}") from exc
    except TimeoutError as exc:
        raise EvaluationError("调用模型 API 超时") from exc

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise EvaluationError("模型 API 返回了无法解析的响应") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError("模型 API 响应不是 JSON 对象")
    return decoded


class DeepSeekEvaluator:
    def __init__(
        self,
        http_post: HttpPost | None = None,
        model_a: str | None = None,
        model_b: str | None = None,
        model_b_api_key: str | None = None,
        model_b_api_url: str | None = None,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EvaluationError(
                "未检测到环境变量 DEEPSEEK_API_KEY。请先配置该环境变量后再运行。"
            )
        self._api_key = api_key
        self._http_post = http_post or _default_http_post
        self.model_a = model_a or os.getenv("EVALFORGE_MODEL_A", MODEL_A)
        self.model_b = model_b or os.getenv("EVALFORGE_MODEL_B", MODEL_B)
        self._model_b_api_key = model_b_api_key or os.getenv(
            "EVALFORGE_MODEL_B_API_KEY", ""
        )
        self._model_b_api_url = model_b_api_url or os.getenv(
            "EVALFORGE_MODEL_B_API_URL", ""
        )

    def evaluate(
        self, question: str, reference_answer: str, candidate_answer: str
    ) -> EvaluationResult:
        base_prompt = (
            "请按既定标准评测以下答案，并返回 JSON。\n"
            f"问题：{question}\n"
            f"参考答案：{reference_answer}\n"
            f"待评估答案：{candidate_answer}"
        )
        history: list[EvaluationAttempt] = []
        first_report = self._call_and_review(
            model=self.model_a,
            stage="model_a_initial",
            user_prompt=base_prompt,
            api_url=API_URL,
            api_key=self._api_key,
            include_thinking=True,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        history.append(self._attempt_from_report("model_a_initial", self.model_a, first_report))
        if first_report.passed:
            return self._build_result(
                first_report,
                needs_review=False,
                model=self.model_a,
                stage="model_a_initial",
                history=history,
            )

        correction_prompt = (
            base_prompt
            + "\n\n上一次输出未通过 Python 复核："
            + "；".join(first_report.issues)
            + "。请根据违规原因重新评估并修正输出。这是模型 A 唯一一次纠正机会。"
        )
        corrected_report = self._call_and_review(
            model=self.model_a,
            stage="model_a_correction",
            user_prompt=correction_prompt,
            api_url=API_URL,
            api_key=self._api_key,
            include_thinking=True,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        history.append(
            self._attempt_from_report(
                "model_a_correction", self.model_a, corrected_report
            )
        )
        if corrected_report.passed:
            return self._build_result(
                corrected_report,
                needs_review=False,
                model=self.model_a,
                stage="model_a_correction",
                history=history,
            )

        # Blind review: model B receives only the original evaluation material.
        self._ensure_model_b_configured()
        blind_report = self._call_and_review(
            model=self.model_b,
            stage="model_b_blind",
            user_prompt=base_prompt,
            api_url=self._model_b_api_url,
            api_key=self._model_b_api_key,
            include_thinking=False,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        history.append(
            self._attempt_from_report("model_b_blind", self.model_b, blind_report)
        )
        return self._build_result(
            blind_report,
            needs_review=not blind_report.passed,
            model=self.model_b,
            stage="model_b_blind",
            history=history,
        )

    def _call_and_review(
        self,
        *,
        model: str,
        stage: str,
        user_prompt: str,
        api_url: str,
        api_key: str,
        include_thinking: bool,
        reference_answer: str,
        candidate_answer: str,
    ) -> ReviewReport:
        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }
            if include_thinking:
                payload["thinking"] = {"type": "disabled"}
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            response = self._http_post(
                api_url, headers, body, REQUEST_TIMEOUT_SECONDS
            )
            content = self._extract_api_content(response)
        except EvaluationError as exc:
            raise EvaluationError(f"{stage} 调用失败：{exc}") from exc
        return review_content(content, reference_answer, candidate_answer)

    def _ensure_model_b_configured(self) -> None:
        missing: list[str] = []
        if not self.model_b:
            missing.append("EVALFORGE_MODEL_B")
        if not self._model_b_api_url:
            missing.append("EVALFORGE_MODEL_B_API_URL")
        if not self._model_b_api_key:
            missing.append("EVALFORGE_MODEL_B_API_KEY")
        if missing:
            raise EvaluationError(
                "模型 A 纠正后仍未通过，但模型 B 尚未配置：" + ", ".join(missing)
            )

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
        report: ReviewReport,
        *,
        needs_review: bool,
        model: str,
        stage: str,
        history: list[EvaluationAttempt],
    ) -> EvaluationResult:
        reason = report.reason or "LLM 输出未通过 Python 复核，需人工处理。"
        return EvaluationResult(
            score=report.score,
            reason=reason,
            keywords=report.keywords,
            covered_keywords=report.covered_keywords,
            coverage_ratio=report.coverage_ratio,
            needs_review=needs_review,
            review_issues=report.issues,
            attempts=len(history),
            model=model,
            stage=stage,
            review_history=tuple(history),
        )

    @staticmethod
    def _attempt_from_report(
        stage: str, model: str, report: ReviewReport
    ) -> EvaluationAttempt:
        return EvaluationAttempt(
            stage=stage,
            model=model,
            passed=report.passed,
            score=report.score,
            issues=report.issues,
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
