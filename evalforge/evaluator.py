from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .reviewer import (
    ExtraClaim,
    KeyPoint,
    KeyPointJudgment,
    ReferenceAnalysis,
    ReviewReport,
    find_judge_disagreement,
    review_content,
    review_reference_content,
    review_result,
)


API_URL = "https://api.deepseek.com/chat/completions"
MODEL_A = "deepseek-v4-flash"
MODEL_B = "glm-5.1"
MODEL_B_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = MODEL_A
TEMPERATURE = 0
MAX_TOKENS = 1500
REQUEST_TIMEOUT_SECONDS = 90
REFERENCE_MAX_ATTEMPTS = 2

REFERENCE_SYSTEM_PROMPT = """你负责把参考答案拆分为原子知识点。每个知识点只能表达一个可验证事实，不得重复，不得加入参考答案没有支持的内容。

importance 只能是 core 或 supporting：core 缺失或错误会改变主要结论；supporting 有助于完整回答，但缺失不推翻核心结论。至少需要一个 core。

如果参考答案存在明显歧义、矛盾、事实错误或不足以支持评分，设置 reference_defect=true 并说明原因；不要强行生成正常知识点。

只返回 JSON，不要输出 Markdown 或额外文字：
{
  "question_id": "输入的问题编号",
  "reference_defect": false,
  "reference_defect_reasons": [],
  "key_points": [
    {"id": "K1", "statement": "单个原子事实", "importance": "core"}
  ]
}"""

SCORE_SYSTEM_PROMPT = """你是严格、稳定的答案质量评测器。依据问题、参考答案和已给定的原子知识点评估待评答案。只判断事实、语义、完整性和相关性，不因同义词、改写、语序、简称、篇幅或文风不同扣分。

每个知识点必须选择一个状态：
- matched：完整表达，允许同义改写；必须引用待评答案原文。
- partial：有所表达但不完整或轻微不精确；必须引用原文。
- missing：没有涉及；evidence 必须为 null。
- contradicted：明确相反或冲突；必须引用冲突原文。

额外陈述 status 只能是 correct_relevant、irrelevant、incorrect、unsupported；severity 只有错误陈述可用 minor 或 major，其余必须为 none。参考答案不是世界知识的穷尽集合，参考答案之外的正确信息不得自动扣分。

评分：5=所有知识点完整正确且无实质错误；4=所有核心点正确，仅补充点轻微遗漏；3=至少一个核心点正确但缺少重要信息；2=少量正确，主要不完整或错误；1=主要结论错误且仅有零散正确信息；0=完全错误、答非所问、拒答或无意义。

先单独判断待评答案自身是否包含不能同时成立的陈述。若存在，不论最终分数高低，都必须设置 candidate_self_contradiction=true、uncertain=true，并给出具体原因；不要把“与参考答案冲突”误当成“答案自身冲突”。参考答案不足或无法确定最终立场时也设置 uncertain=true。

只返回 JSON，不要输出 Markdown 或额外文字，字段必须严格如下：
{
  "question_id": "输入的问题编号",
  "key_point_judgments": [
    {"id": "K1", "status": "matched", "evidence": "待评答案原文", "explanation": "语义判断"}
  ],
  "extra_claims": [
    {"claim": "额外陈述", "status": "incorrect", "evidence": "待评答案原文", "severity": "major", "explanation": "判断理由"}
  ],
  "candidate_self_contradiction": false,
  "score": 4,
  "reason": "综合评分理由",
  "uncertain": false,
  "uncertainty_reasons": []
}"""


class EvaluationError(RuntimeError):
    """Raised when an API or configuration error prevents evaluation."""


@dataclass(frozen=True)
class EvaluationAttempt:
    stage: str
    model: str
    passed: bool
    score: int | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    duration_ms: int
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_result: str = ""


@dataclass(frozen=True)
class PreparedReference:
    analysis: ReferenceAnalysis
    attempts: tuple[EvaluationAttempt, ...]


@dataclass(frozen=True)
class EvaluationResult:
    score: int | None
    reason: str
    key_point_judgments: tuple[KeyPointJudgment, ...] = ()
    extra_claims: tuple[ExtraClaim, ...] = ()
    candidate_self_contradiction: bool = False
    semantic_coverage: float = 0.0
    lexical_overlap: float = 0.0
    needs_review: bool = False
    review_issues: tuple[str, ...] = ()
    validation_warnings: tuple[str, ...] = ()
    attempts: int = 0
    model: str = MODEL_A
    stage: str = "model_a_initial"
    status: str = "accepted_model_a"
    review_history: tuple[EvaluationAttempt, ...] = ()
    reference_analysis: ReferenceAnalysis | None = None
    reference_history: tuple[EvaluationAttempt, ...] = ()


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
        self.model_a = model_a or os.getenv("EVALFORGE_MODEL_A") or MODEL_A
        self.model_b = model_b or os.getenv("EVALFORGE_MODEL_B") or MODEL_B
        self._model_b_api_key = (
            model_b_api_key
            or os.getenv("EVALFORGE_MODEL_B_API_KEY")
            or os.getenv("GLM_API_KEY", "")
        )
        self._model_b_api_url = (
            model_b_api_url
            or os.getenv("EVALFORGE_MODEL_B_API_URL")
            or MODEL_B_API_URL
        )
        self._reference_cache: dict[tuple[str, str, str], PreparedReference] = {}

    def prepare_reference(
        self, question_id: str, question: str, reference_answer: str
    ) -> PreparedReference:
        cache_key = (question_id, question, reference_answer)
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return cached

        base_prompt = (
            "请原子化以下参考答案。\n"
            f"问题编号：{question_id}\n"
            f"问题：{question}\n"
            f"参考答案：{reference_answer}"
        )
        feedback = ""
        attempts: list[EvaluationAttempt] = []
        for attempt_number in range(1, REFERENCE_MAX_ATTEMPTS + 1):
            content, duration_ms, usage = self._request_content(
                api_url=API_URL,
                api_key=self._api_key,
                model=self.model_a,
                system_prompt=REFERENCE_SYSTEM_PROMPT,
                user_prompt=base_prompt + feedback,
                stage=f"reference_atomization_{attempt_number}",
            )
            report = review_reference_content(content, question_id)
            attempts.append(
                EvaluationAttempt(
                    stage=f"reference_atomization_{attempt_number}",
                    model=self.model_a,
                    passed=report.passed,
                    score=None,
                    issues=report.issues,
                    warnings=(),
                    duration_ms=duration_ms,
                    token_usage=usage,
                    raw_result=content,
                )
            )
            if report.passed and report.analysis is not None:
                prepared = PreparedReference(
                    analysis=report.analysis,
                    attempts=tuple(attempts),
                )
                self._reference_cache[cache_key] = prepared
                return prepared
            if attempt_number < REFERENCE_MAX_ATTEMPTS:
                feedback = (
                    "\n\n上一次原子化结果未通过 Python 校验："
                    + "；".join(report.issues)
                    + "。请修正 JSON。"
                )
        raise EvaluationError(
            "参考答案原子化连续失败：" + "；".join(attempts[-1].issues)
        )

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        question_id: str = "Q",
    ) -> EvaluationResult:
        prepared = self.prepare_reference(question_id, question, reference_answer)
        analysis = prepared.analysis
        if analysis.reference_defect:
            reasons = tuple(
                f"reference_defect: {reason}"
                for reason in analysis.reference_defect_reasons
            ) or ("reference_defect: 参考答案被标记为存在缺陷",)
            return EvaluationResult(
                score=None,
                reason="参考答案存在缺陷，无法自动评分。",
                needs_review=True,
                review_issues=reasons,
                model=self.model_a,
                stage="reference_atomization",
                status="reference_defect",
                reference_analysis=analysis,
                reference_history=prepared.attempts,
            )

        key_points_json = json.dumps(
            [
                {
                    "id": point.id,
                    "statement": point.statement,
                    "importance": point.importance,
                }
                for point in analysis.key_points
            ],
            ensure_ascii=False,
        )
        base_prompt = (
            "请评测以下待评答案。\n"
            f"问题编号：{question_id}\n"
            f"问题：{question}\n"
            f"参考答案：{reference_answer}\n"
            f"原子知识点：{key_points_json}\n"
            f"待评答案：{candidate_answer}"
        )

        history: list[EvaluationAttempt] = []
        first_report, first_attempt = self._call_and_review(
            model=self.model_a,
            stage="model_a_initial",
            user_prompt=base_prompt,
            api_url=API_URL,
            api_key=self._api_key,
            question_id=question_id,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            key_points=analysis.key_points,
        )
        history.append(first_attempt)
        if first_report.acceptable:
            return self._build_result(
                first_report,
                model=self.model_a,
                stage="model_a_initial",
                status="accepted_model_a",
                history=history,
                prepared=prepared,
            )

        correction_reasons = list(first_report.issues)
        if first_report.uncertain:
            correction_reasons.append(
                "judge_uncertain: " + "；".join(first_report.uncertainty_reasons)
            )
        correction_prompt = (
            base_prompt
            + "\n\n上一次输出未通过 Python 硬校验或存在不确定性："
            + "；".join(correction_reasons)
            + "。请只针对这些可验证问题纠正一次。"
        )
        corrected_report, corrected_attempt = self._call_and_review(
            model=self.model_a,
            stage="model_a_correction",
            user_prompt=correction_prompt,
            api_url=API_URL,
            api_key=self._api_key,
            question_id=question_id,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            key_points=analysis.key_points,
        )
        history.append(corrected_attempt)
        if corrected_report.acceptable:
            return self._build_result(
                corrected_report,
                model=self.model_a,
                stage="model_a_correction",
                status="accepted_model_a_retry",
                history=history,
                prepared=prepared,
            )

        self._ensure_model_b_configured()
        blind_report, blind_attempt = self._call_and_review(
            model=self.model_b,
            stage="model_b_blind",
            user_prompt=base_prompt,
            api_url=self._model_b_api_url,
            api_key=self._model_b_api_key,
            question_id=question_id,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            key_points=analysis.key_points,
        )
        history.append(blind_attempt)

        review_reasons: list[str] = []
        if not blind_report.passed:
            review_reasons.extend(blind_report.issues)
        if blind_report.uncertain:
            review_reasons.extend(
                f"judge_uncertain: {reason}"
                for reason in blind_report.uncertainty_reasons
            )
        if blind_report.acceptable and corrected_report.passed:
            review_reasons.extend(
                find_judge_disagreement(
                    corrected_report, blind_report, analysis.key_points
                )
            )

        if review_reasons:
            return self._build_result(
                blind_report,
                model=self.model_b,
                stage="model_b_blind",
                status="needs_human_review",
                history=history,
                prepared=prepared,
                needs_review=True,
                extra_review_issues=tuple(review_reasons),
            )
        return self._build_result(
            blind_report,
            model=self.model_b,
            stage="model_b_blind",
            status="accepted_model_b",
            history=history,
            prepared=prepared,
        )

    def _call_and_review(
        self,
        *,
        model: str,
        stage: str,
        user_prompt: str,
        api_url: str,
        api_key: str,
        question_id: str,
        reference_answer: str,
        candidate_answer: str,
        key_points: tuple[KeyPoint, ...],
    ) -> tuple[ReviewReport, EvaluationAttempt]:
        content, duration_ms, usage = self._request_content(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=SCORE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            stage=stage,
        )
        report = review_content(
            content,
            question_id,
            reference_answer,
            candidate_answer,
            key_points,
        )
        attempt = EvaluationAttempt(
            stage=stage,
            model=model,
            passed=report.acceptable,
            score=report.score,
            issues=report.issues
            + (
                tuple(
                    f"judge_uncertain: {reason}"
                    for reason in report.uncertainty_reasons
                )
                if report.uncertain
                else ()
            ),
            warnings=report.warnings,
            duration_ms=duration_ms,
            token_usage=usage,
            raw_result=content,
        )
        return report, attempt

    def _request_content(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> tuple[str, int, dict[str, int]]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        try:
            response = self._http_post(
                api_url, headers, body, REQUEST_TIMEOUT_SECONDS
            )
            content = self._extract_api_content(response)
        except EvaluationError as exc:
            raise EvaluationError(f"{stage} 调用失败：{exc}") from exc
        duration_ms = round((time.perf_counter() - started) * 1000)
        return content, duration_ms, self._extract_usage(response)

    def _ensure_model_b_configured(self) -> None:
        if not self._model_b_api_key:
            raise EvaluationError(
                "模型 A 纠正后仍未通过，但模型 B 尚未配置："
                "GLM_API_KEY（或 EVALFORGE_MODEL_B_API_KEY）"
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
    def _extract_usage(response: dict[str, Any]) -> dict[str, int]:
        raw_usage = response.get("usage")
        if not isinstance(raw_usage, dict):
            return {}
        return {
            key: value
            for key, value in raw_usage.items()
            if isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
        }

    @staticmethod
    def _build_result(
        report: ReviewReport,
        *,
        model: str,
        stage: str,
        status: str,
        history: list[EvaluationAttempt],
        prepared: PreparedReference,
        needs_review: bool = False,
        extra_review_issues: tuple[str, ...] = (),
    ) -> EvaluationResult:
        review_issues = extra_review_issues or report.issues
        reason = report.reason or "模型输出未通过 Python 硬校验，需人工处理。"
        return EvaluationResult(
            score=report.score,
            reason=reason,
            key_point_judgments=report.key_point_judgments,
            extra_claims=report.extra_claims,
            candidate_self_contradiction=report.candidate_self_contradiction,
            semantic_coverage=report.semantic_coverage,
            lexical_overlap=report.lexical_overlap,
            needs_review=needs_review,
            review_issues=review_issues,
            validation_warnings=report.warnings,
            attempts=len(history),
            model=model,
            stage=stage,
            status=status,
            review_history=tuple(history),
            reference_analysis=prepared.analysis,
            reference_history=prepared.attempts,
        )


def validate_result(
    result: Any,
    question_id: str,
    reference_answer: str,
    candidate_answer: str,
    key_points: tuple[KeyPoint, ...],
) -> EvaluationResult:
    report = review_result(
        result,
        question_id,
        reference_answer,
        candidate_answer,
        key_points,
    )
    if report.issues:
        raise EvaluationError("；".join(report.issues))
    return EvaluationResult(
        score=report.score,
        reason=report.reason,
        key_point_judgments=report.key_point_judgments,
        extra_claims=report.extra_claims,
        candidate_self_contradiction=report.candidate_self_contradiction,
        semantic_coverage=report.semantic_coverage,
        lexical_overlap=report.lexical_overlap,
        validation_warnings=report.warnings,
    )
