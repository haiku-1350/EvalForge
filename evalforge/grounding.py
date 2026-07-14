from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


GROUNDING_FIELDS = {
    "question_id",
    "claims",
    "groundedness_score",
    "reason",
    "uncertain",
    "uncertainty_reasons",
}
CLAIM_FIELDS = {
    "claim",
    "status",
    "answer_evidence",
    "context_evidence",
    "explanation",
}
CLAIM_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported",
    "non_factual",
}


@dataclass(frozen=True)
class GroundingClaim:
    claim: str
    status: str
    answer_evidence: str
    context_evidence: str | None
    explanation: str


@dataclass(frozen=True)
class GroundingReport:
    question_id: str
    score: int | None
    reason: str
    claims: tuple[GroundingClaim, ...]
    uncertain: bool
    uncertainty_reasons: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def acceptable(self) -> bool:
        return self.passed and not self.uncertain


def review_grounding_content(
    content: str,
    question_id: str,
    retrieved_context: str,
    answer: str,
) -> GroundingReport:
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _empty_report(question_id, ("invalid_json: groundedness 输出不是合法 JSON",))
    return review_grounding_result(result, question_id, retrieved_context, answer)


def review_grounding_result(
    result: Any,
    question_id: str,
    retrieved_context: str,
    answer: str,
) -> GroundingReport:
    if not isinstance(result, dict):
        return _empty_report(question_id, ("schema_invalid: groundedness 结果必须是 JSON 对象",))

    issues: list[str] = []
    fields = set(result)
    missing = sorted(GROUNDING_FIELDS - fields)
    extra = sorted(fields - GROUNDING_FIELDS)
    if missing:
        issues.append("schema_invalid: 缺少字段 " + ", ".join(missing))
    if extra:
        issues.append("schema_invalid: 包含多余字段 " + ", ".join(extra))
    if result.get("question_id") != question_id:
        issues.append("schema_invalid: question_id 与输入不一致")

    raw_score = result.get("groundedness_score")
    score: int | None = None
    if isinstance(raw_score, bool) or not isinstance(raw_score, int):
        issues.append("score_out_of_range: groundedness_score 必须是 0 到 5 的整数")
    elif not 0 <= raw_score <= 5:
        issues.append("score_out_of_range: groundedness_score 必须在 0 到 5 之间")
    else:
        score = raw_score

    raw_reason = result.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if not reason:
        issues.append("schema_invalid: reason 必须是非空字符串")

    raw_uncertain = result.get("uncertain")
    uncertain = raw_uncertain if isinstance(raw_uncertain, bool) else False
    if not isinstance(raw_uncertain, bool):
        issues.append("schema_invalid: uncertain 必须是布尔值")

    raw_uncertainty_reasons = result.get("uncertainty_reasons")
    uncertainty_reasons: tuple[str, ...] = ()
    if not isinstance(raw_uncertainty_reasons, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in raw_uncertainty_reasons
    ):
        issues.append("schema_invalid: uncertainty_reasons 必须是非空字符串数组")
    else:
        uncertainty_reasons = tuple(item.strip() for item in raw_uncertainty_reasons)
    if uncertain and not uncertainty_reasons:
        issues.append("judge_uncertain: uncertain=true 时必须提供原因")

    claims = _review_claims(result.get("claims"), retrieved_context, answer, issues)
    if score is not None and claims:
        _check_score_conflicts(score, claims, issues)

    return GroundingReport(
        question_id=question_id,
        score=score,
        reason=reason,
        claims=tuple(claims),
        uncertain=uncertain,
        uncertainty_reasons=uncertainty_reasons,
        issues=tuple(issues),
    )


def _review_claims(
    raw_claims: Any,
    retrieved_context: str,
    answer: str,
    issues: list[str],
) -> list[GroundingClaim]:
    if not isinstance(raw_claims, list) or not raw_claims:
        issues.append("schema_invalid: claims 必须是非空数组")
        return []
    claims: list[GroundingClaim] = []
    for index, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict) or set(raw) != CLAIM_FIELDS:
            issues.append(f"schema_invalid: 第 {index} 个 claim 字段不正确")
            continue
        claim = raw.get("claim")
        status = raw.get("status")
        answer_evidence = raw.get("answer_evidence")
        context_evidence = raw.get("context_evidence")
        explanation = raw.get("explanation")
        if not isinstance(claim, str) or not claim.strip():
            issues.append(f"schema_invalid: 第 {index} 个 claim 不能为空")
            continue
        if status not in CLAIM_STATUSES:
            issues.append(f"schema_invalid: 第 {index} 个 claim status 非法")
            continue
        if not isinstance(answer_evidence, str) or not answer_evidence.strip():
            issues.append(f"schema_invalid: 第 {index} 个 answer_evidence 不能为空")
            continue
        if answer_evidence not in answer:
            issues.append(f"evidence_not_found: 第 {index} 个 answer_evidence 不在回答原文中")
        if not isinstance(explanation, str) or not explanation.strip():
            issues.append(f"schema_invalid: 第 {index} 个 explanation 不能为空")
            continue

        if status in {"supported", "partially_supported"}:
            if not isinstance(context_evidence, str) or not context_evidence.strip():
                issues.append(
                    f"schema_invalid: 第 {index} 个受支持 claim 必须提供检索原文证据"
                )
                continue
            if context_evidence not in retrieved_context:
                issues.append(
                    f"evidence_not_found: 第 {index} 个 context_evidence 不在检索内容中"
                )
        elif context_evidence is not None:
            issues.append(
                f"schema_invalid: 第 {index} 个 {status} claim 的 context_evidence 必须为 null"
            )
            continue

        claims.append(
            GroundingClaim(
                claim=claim.strip(),
                status=status,
                answer_evidence=answer_evidence.strip(),
                context_evidence=context_evidence.strip()
                if isinstance(context_evidence, str)
                else None,
                explanation=explanation.strip(),
            )
        )
    return claims


def _check_score_conflicts(
    score: int,
    claims: list[GroundingClaim],
    issues: list[str],
) -> None:
    factual = [claim for claim in claims if claim.status != "non_factual"]
    if score == 5 and any(claim.status != "supported" for claim in factual):
        issues.append("score_state_conflict: 5 分不能包含部分支持或无支持的事实陈述")
    if score >= 4 and any(claim.status == "unsupported" for claim in factual):
        issues.append("score_state_conflict: 4 至 5 分不能包含无支持的事实陈述")
    if score == 0 and any(claim.status == "supported" for claim in factual):
        issues.append("score_state_conflict: 0 分不能包含有检索支持的事实陈述")


def _empty_report(question_id: str, issues: tuple[str, ...]) -> GroundingReport:
    return GroundingReport(
        question_id=question_id,
        score=None,
        reason="",
        claims=(),
        uncertain=False,
        uncertainty_reasons=(),
        issues=issues,
    )
