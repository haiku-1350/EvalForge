from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


IMPORTANCE_VALUES = {"core", "supporting"}
JUDGMENT_STATUSES = {"matched", "partial", "missing", "contradicted"}
EXTRA_CLAIM_STATUSES = {
    "correct_relevant",
    "irrelevant",
    "incorrect",
    "unsupported",
}
EXTRA_CLAIM_SEVERITIES = {"none", "minor", "major"}
REFERENCE_FIELDS = {
    "question_id",
    "reference_defect",
    "reference_defect_reasons",
    "key_points",
}
RESULT_FIELDS = {
    "question_id",
    "key_point_judgments",
    "extra_claims",
    "candidate_self_contradiction",
    "score",
    "reason",
    "uncertain",
    "uncertainty_reasons",
}


@dataclass(frozen=True)
class KeyPoint:
    id: str
    statement: str
    importance: str


@dataclass(frozen=True)
class ReferenceAnalysis:
    question_id: str
    key_points: tuple[KeyPoint, ...]
    reference_defect: bool
    reference_defect_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceReviewReport:
    analysis: ReferenceAnalysis | None
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues and self.analysis is not None


@dataclass(frozen=True)
class KeyPointJudgment:
    id: str
    status: str
    evidence: str | None
    explanation: str


@dataclass(frozen=True)
class ExtraClaim:
    claim: str
    status: str
    evidence: str
    severity: str
    explanation: str


@dataclass(frozen=True)
class ReviewReport:
    question_id: str
    score: int | None
    reason: str
    key_point_judgments: tuple[KeyPointJudgment, ...]
    extra_claims: tuple[ExtraClaim, ...]
    candidate_self_contradiction: bool
    uncertain: bool
    uncertainty_reasons: tuple[str, ...]
    semantic_coverage: float
    lexical_overlap: float
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def acceptable(self) -> bool:
        return self.passed and not self.uncertain

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.split(":", 1)[0] for issue in self.issues)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _lexical_overlap(reference_answer: str, candidate_answer: str) -> float:
    reference = _normalize(reference_answer)
    candidate = _normalize(candidate_answer)
    if not reference or not candidate:
        return 0.0
    return SequenceMatcher(None, reference, candidate).ratio()


def _parse_json(content: str, label: str) -> tuple[Any | None, tuple[str, ...]]:
    try:
        return json.loads(content), ()
    except (json.JSONDecodeError, TypeError):
        return None, (f"invalid_json: {label}不是合法 JSON",)


def review_reference_content(content: str, question_id: str) -> ReferenceReviewReport:
    result, parse_issues = _parse_json(content, "参考答案原子化输出")
    if parse_issues:
        return ReferenceReviewReport(analysis=None, issues=parse_issues)
    if not isinstance(result, dict):
        return ReferenceReviewReport(
            analysis=None,
            issues=("schema_invalid: 参考答案原子化结果必须是 JSON 对象",),
        )

    issues: list[str] = []
    fields = set(result)
    missing_fields = sorted(REFERENCE_FIELDS - fields)
    extra_fields = sorted(fields - REFERENCE_FIELDS)
    if missing_fields:
        issues.append("schema_invalid: 缺少字段 " + ", ".join(missing_fields))
    if extra_fields:
        issues.append("schema_invalid: 包含多余字段 " + ", ".join(extra_fields))

    raw_question_id = result.get("question_id")
    if raw_question_id != question_id:
        issues.append("schema_invalid: question_id 与输入不一致")

    raw_defect = result.get("reference_defect")
    reference_defect = raw_defect if isinstance(raw_defect, bool) else False
    if not isinstance(raw_defect, bool):
        issues.append("schema_invalid: reference_defect 必须是布尔值")

    raw_reasons = result.get("reference_defect_reasons")
    defect_reasons: tuple[str, ...] = ()
    if not isinstance(raw_reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in raw_reasons
    ):
        issues.append("schema_invalid: reference_defect_reasons 必须是非空字符串数组")
    else:
        defect_reasons = tuple(reason.strip() for reason in raw_reasons)
    if reference_defect and not defect_reasons:
        issues.append("reference_defect: 标记参考答案缺陷时必须给出原因")

    raw_key_points = result.get("key_points")
    key_points: list[KeyPoint] = []
    if not isinstance(raw_key_points, list):
        issues.append("schema_invalid: key_points 必须是数组")
    else:
        seen_ids: set[str] = set()
        for index, raw_point in enumerate(raw_key_points, start=1):
            if not isinstance(raw_point, dict):
                issues.append(f"schema_invalid: 第 {index} 个知识点必须是对象")
                continue
            if set(raw_point) != {"id", "statement", "importance"}:
                issues.append(
                    f"schema_invalid: 第 {index} 个知识点字段必须为 id、statement、importance"
                )
                continue
            point_id = raw_point.get("id")
            statement = raw_point.get("statement")
            importance = raw_point.get("importance")
            if not isinstance(point_id, str) or not point_id.strip():
                issues.append(f"schema_invalid: 第 {index} 个知识点 id 非法")
                continue
            point_id = point_id.strip()
            if point_id in seen_ids:
                issues.append(f"schema_invalid: 知识点 id 重复：{point_id}")
                continue
            seen_ids.add(point_id)
            if not isinstance(statement, str) or not statement.strip():
                issues.append(f"schema_invalid: 知识点 {point_id} statement 非法")
                continue
            if importance not in IMPORTANCE_VALUES:
                issues.append(f"schema_invalid: 知识点 {point_id} importance 非法")
                continue
            key_points.append(
                KeyPoint(
                    id=point_id,
                    statement=statement.strip(),
                    importance=importance,
                )
            )

    if not reference_defect:
        if not key_points:
            issues.append("schema_invalid: 有效参考答案至少需要一个知识点")
        if key_points and not any(point.importance == "core" for point in key_points):
            issues.append("schema_invalid: 至少需要一个 core 知识点")

    analysis = ReferenceAnalysis(
        question_id=question_id,
        key_points=tuple(key_points),
        reference_defect=reference_defect,
        reference_defect_reasons=defect_reasons,
    )
    return ReferenceReviewReport(analysis=analysis, issues=tuple(issues))


def review_content(
    content: str,
    question_id: str,
    reference_answer: str,
    candidate_answer: str,
    key_points: tuple[KeyPoint, ...],
) -> ReviewReport:
    result, parse_issues = _parse_json(content, "评分输出")
    if parse_issues:
        return _empty_report(
            question_id,
            reference_answer,
            candidate_answer,
            parse_issues,
        )
    return review_result(
        result,
        question_id,
        reference_answer,
        candidate_answer,
        key_points,
    )


def review_result(
    result: Any,
    question_id: str,
    reference_answer: str,
    candidate_answer: str,
    key_points: tuple[KeyPoint, ...],
) -> ReviewReport:
    lexical_overlap = _lexical_overlap(reference_answer, candidate_answer)
    if not isinstance(result, dict):
        return _empty_report(
            question_id,
            reference_answer,
            candidate_answer,
            ("schema_invalid: 评分结果必须是 JSON 对象",),
        )

    issues: list[str] = []
    fields = set(result)
    missing_fields = sorted(RESULT_FIELDS - fields)
    extra_fields = sorted(fields - RESULT_FIELDS)
    if missing_fields:
        issues.append("schema_invalid: 缺少字段 " + ", ".join(missing_fields))
    if extra_fields:
        issues.append("schema_invalid: 包含多余字段 " + ", ".join(extra_fields))
    if result.get("question_id") != question_id:
        issues.append("schema_invalid: question_id 与输入不一致")

    raw_score = result.get("score")
    score: int | None = None
    if isinstance(raw_score, bool) or not isinstance(raw_score, int):
        issues.append("score_out_of_range: score 必须是 0 到 5 的整数")
    elif not 0 <= raw_score <= 5:
        issues.append("score_out_of_range: score 必须在 0 到 5 之间")
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

    raw_self_contradiction = result.get("candidate_self_contradiction")
    candidate_self_contradiction = (
        raw_self_contradiction if isinstance(raw_self_contradiction, bool) else False
    )
    if not isinstance(raw_self_contradiction, bool):
        issues.append("schema_invalid: candidate_self_contradiction 必须是布尔值")

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
    if candidate_self_contradiction and not uncertain:
        issues.append(
            "self_contradiction_conflict: 待评答案内部自相矛盾时必须设置 uncertain=true"
        )

    judgments = _review_judgments(
        result.get("key_point_judgments"), key_points, candidate_answer, issues
    )
    extra_claims = _review_extra_claims(
        result.get("extra_claims"), candidate_answer, issues
    )
    semantic_coverage = _semantic_coverage(judgments, key_points)

    if score is not None and len(judgments) == len(key_points):
        _check_score_state_conflicts(score, judgments, extra_claims, key_points, issues)

    warnings: list[str] = []
    if semantic_coverage >= 0.75 and lexical_overlap < 0.5:
        warnings.append("lexical_semantic_mismatch")

    return ReviewReport(
        question_id=question_id,
        score=score,
        reason=reason,
        key_point_judgments=tuple(judgments),
        extra_claims=tuple(extra_claims),
        candidate_self_contradiction=candidate_self_contradiction,
        uncertain=uncertain,
        uncertainty_reasons=uncertainty_reasons,
        semantic_coverage=semantic_coverage,
        lexical_overlap=lexical_overlap,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _review_judgments(
    raw_judgments: Any,
    key_points: tuple[KeyPoint, ...],
    candidate_answer: str,
    issues: list[str],
) -> list[KeyPointJudgment]:
    if not isinstance(raw_judgments, list):
        issues.append("schema_invalid: key_point_judgments 必须是数组")
        return []

    expected = {point.id for point in key_points}
    seen: set[str] = set()
    judgments: list[KeyPointJudgment] = []
    for index, raw in enumerate(raw_judgments, start=1):
        if not isinstance(raw, dict):
            issues.append(f"schema_invalid: 第 {index} 个知识点判断必须是对象")
            continue
        if set(raw) != {"id", "status", "evidence", "explanation"}:
            issues.append(f"schema_invalid: 第 {index} 个知识点判断字段不正确")
            continue
        point_id = raw.get("id")
        status = raw.get("status")
        evidence = raw.get("evidence")
        explanation = raw.get("explanation")
        if not isinstance(point_id, str) or point_id not in expected:
            issues.append(f"unexpected_key_point_id: 非法知识点 id：{point_id}")
            continue
        if point_id in seen:
            issues.append(f"duplicate_key_point_id: 知识点 id 重复：{point_id}")
            continue
        seen.add(point_id)
        if status not in JUDGMENT_STATUSES:
            issues.append(f"schema_invalid: 知识点 {point_id} status 非法")
            continue
        if not isinstance(explanation, str) or not explanation.strip():
            issues.append(f"schema_invalid: 知识点 {point_id} explanation 不能为空")
            explanation = ""
        if status == "missing":
            if evidence is not None:
                issues.append(f"schema_invalid: missing 状态的 {point_id} evidence 必须为 null")
                evidence = None
        else:
            if not isinstance(evidence, str) or not evidence.strip():
                issues.append(f"schema_invalid: {status} 状态的 {point_id} 必须提供 evidence")
                evidence = None
            else:
                evidence = evidence.strip()
                if evidence not in candidate_answer:
                    issues.append(
                        f"evidence_not_found: 知识点 {point_id} 的 evidence 不在待评答案中"
                    )
        judgments.append(
            KeyPointJudgment(
                id=point_id,
                status=status,
                evidence=evidence,
                explanation=explanation.strip() if isinstance(explanation, str) else "",
            )
        )

    missing_ids = sorted(expected - seen)
    if missing_ids:
        issues.append("missing_key_point_id: 缺少知识点判断 " + ", ".join(missing_ids))
    return judgments


def _review_extra_claims(
    raw_claims: Any, candidate_answer: str, issues: list[str]
) -> list[ExtraClaim]:
    if not isinstance(raw_claims, list):
        issues.append("schema_invalid: extra_claims 必须是数组")
        return []
    claims: list[ExtraClaim] = []
    for index, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict) or set(raw) != {
            "claim",
            "status",
            "evidence",
            "severity",
            "explanation",
        }:
            issues.append(f"schema_invalid: 第 {index} 个额外陈述字段不正确")
            continue
        claim = raw.get("claim")
        status = raw.get("status")
        evidence = raw.get("evidence")
        severity = raw.get("severity")
        explanation = raw.get("explanation")
        if not isinstance(claim, str) or not claim.strip():
            issues.append(f"schema_invalid: 第 {index} 个额外陈述 claim 不能为空")
            continue
        if status not in EXTRA_CLAIM_STATUSES:
            issues.append(f"schema_invalid: 第 {index} 个额外陈述 status 非法")
            continue
        if severity not in EXTRA_CLAIM_SEVERITIES:
            issues.append(f"schema_invalid: 第 {index} 个额外陈述 severity 非法")
            continue
        if status == "incorrect" and severity == "none":
            issues.append(f"schema_invalid: incorrect 额外陈述必须标记严重程度")
        if status != "incorrect" and severity != "none":
            issues.append(f"schema_invalid: 非错误额外陈述 severity 必须为 none")
        if not isinstance(evidence, str) or not evidence.strip():
            issues.append(f"schema_invalid: 第 {index} 个额外陈述 evidence 不能为空")
            continue
        evidence = evidence.strip()
        if evidence not in candidate_answer:
            issues.append(f"evidence_not_found: 第 {index} 个额外陈述证据不在待评答案中")
        if not isinstance(explanation, str) or not explanation.strip():
            issues.append(f"schema_invalid: 第 {index} 个额外陈述 explanation 不能为空")
            explanation = ""
        claims.append(
            ExtraClaim(
                claim=claim.strip(),
                status=status,
                evidence=evidence,
                severity=severity,
                explanation=explanation.strip() if isinstance(explanation, str) else "",
            )
        )
    return claims


def _semantic_coverage(
    judgments: list[KeyPointJudgment], key_points: tuple[KeyPoint, ...]
) -> float:
    if not key_points:
        return 0.0
    judgment_by_id = {judgment.id: judgment for judgment in judgments}
    status_factor = {"matched": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
    weighted_total = 0.0
    total_weight = 0.0
    for point in key_points:
        weight = 2.0 if point.importance == "core" else 1.0
        total_weight += weight
        judgment = judgment_by_id.get(point.id)
        if judgment is not None:
            weighted_total += weight * status_factor[judgment.status]
    return weighted_total / total_weight if total_weight else 0.0


def _check_score_state_conflicts(
    score: int,
    judgments: list[KeyPointJudgment],
    extra_claims: list[ExtraClaim],
    key_points: tuple[KeyPoint, ...],
    issues: list[str],
) -> None:
    importance_by_id = {point.id: point.importance for point in key_points}
    if score == 5 and any(judgment.status != "matched" for judgment in judgments):
        issues.append("score_state_conflict: 5 分要求所有知识点均为 matched")
    if score == 4 and any(
        importance_by_id.get(judgment.id) == "core"
        and judgment.status in {"missing", "contradicted"}
        for judgment in judgments
    ):
        issues.append("score_state_conflict: 4 分不能存在缺失或冲突的核心知识点")
    if score >= 4 and any(
        claim.status == "incorrect" and claim.severity == "major"
        for claim in extra_claims
    ):
        issues.append("score_state_conflict: 4 至 5 分不能包含严重错误陈述")
    if score == 0 and any(
        importance_by_id.get(judgment.id) == "core" and judgment.status == "matched"
        for judgment in judgments
    ):
        issues.append("score_state_conflict: 0 分不能包含明确匹配的核心知识点")


def find_judge_disagreement(
    model_a: ReviewReport,
    model_b: ReviewReport,
    key_points: tuple[KeyPoint, ...],
) -> tuple[str, ...]:
    if not model_a.passed or not model_b.passed:
        return ()
    reasons: list[str] = []
    if model_a.score is not None and model_b.score is not None:
        if abs(model_a.score - model_b.score) >= 2:
            reasons.append(
                f"judge_disagreement: 模型分数相差 {abs(model_a.score - model_b.score)} 分"
            )
    core_ids = {point.id for point in key_points if point.importance == "core"}
    a_status = {item.id: item.status for item in model_a.key_point_judgments}
    b_status = {item.id: item.status for item in model_b.key_point_judgments}
    for point_id in sorted(core_ids):
        statuses = {a_status.get(point_id), b_status.get(point_id)}
        if statuses == {"matched", "contradicted"}:
            reasons.append(
                f"judge_disagreement: 核心知识点 {point_id} 出现 matched 与 contradicted 冲突"
            )
    return tuple(reasons)


def _empty_report(
    question_id: str,
    reference_answer: str,
    candidate_answer: str,
    issues: tuple[str, ...],
) -> ReviewReport:
    return ReviewReport(
        question_id=question_id,
        score=None,
        reason="",
        key_point_judgments=(),
        extra_claims=(),
        candidate_self_contradiction=False,
        uncertain=False,
        uncertainty_reasons=(),
        semantic_coverage=0.0,
        lexical_overlap=_lexical_overlap(reference_answer, candidate_answer),
        issues=issues,
        warnings=(),
    )
