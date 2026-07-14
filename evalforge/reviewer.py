from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any


EXPECTED_FIELDS = {"score", "reason", "keywords"}
MIN_KEYWORDS = 1
MAX_KEYWORDS = 8


@dataclass(frozen=True)
class ReviewReport:
    score: int | None
    reason: str
    keywords: tuple[str, ...]
    covered_keywords: tuple[str, ...]
    coverage_ratio: float
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def _normalize(text: str) -> str:
    """Normalize text for deterministic, punctuation-insensitive matching."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _score_coverage_issue(score: int, covered: int, total: int) -> str | None:
    if score <= 1:
        return None
    if score == 2 and covered == 0:
        return "评分冲突：2 分至少需要覆盖 1 个关键词"

    coverage = covered / total
    minimum = {3: 0.5, 4: 0.75, 5: 1.0}.get(score)
    if minimum is not None and coverage < minimum:
        return f"评分冲突：{score} 分要求关键词覆盖率至少为 {minimum:.0%}"
    return None


def review_content(
    content: str, reference_answer: str, candidate_answer: str
) -> ReviewReport:
    """Parse and deterministically review one raw LLM response."""
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return ReviewReport(
            score=None,
            reason="",
            keywords=(),
            covered_keywords=(),
            coverage_ratio=0.0,
            issues=("格式错误：LLM 输出不是合法 JSON",),
        )
    return review_result(result, reference_answer, candidate_answer)


def review_result(
    result: Any, reference_answer: str, candidate_answer: str
) -> ReviewReport:
    """Check schema, score range, keyword validity, coverage, and score consistency."""
    if not isinstance(result, dict):
        return ReviewReport(
            score=None,
            reason="",
            keywords=(),
            covered_keywords=(),
            coverage_ratio=0.0,
            issues=("格式错误：评测结果必须是 JSON 对象",),
        )

    issues: list[str] = []
    fields = set(result)
    missing = sorted(EXPECTED_FIELDS - fields)
    extra = sorted(fields - EXPECTED_FIELDS)
    if missing:
        issues.append(f"格式错误：缺少字段 {', '.join(missing)}")
    if extra:
        issues.append(f"格式错误：包含多余字段 {', '.join(extra)}")

    raw_score = result.get("score")
    score: int | None = None
    if isinstance(raw_score, bool) or not isinstance(raw_score, int):
        issues.append("分数错误：score 必须是整数")
    elif not 0 <= raw_score <= 5:
        issues.append("分数错误：score 必须在 0 到 5 之间")
    else:
        score = raw_score

    raw_reason = result.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if not reason:
        issues.append("格式错误：reason 必须是非空字符串")

    raw_keywords = result.get("keywords")
    keywords: tuple[str, ...] = ()
    keywords_are_valid = True
    if not isinstance(raw_keywords, list):
        issues.append("格式错误：keywords 必须是字符串数组")
        keywords_are_valid = False
    else:
        cleaned: list[str] = []
        for keyword in raw_keywords:
            if not isinstance(keyword, str) or not keyword.strip():
                keywords_are_valid = False
                continue
            cleaned.append(keyword.strip())
        keywords = tuple(cleaned)
        if not MIN_KEYWORDS <= len(raw_keywords) <= MAX_KEYWORDS:
            issues.append(
                f"格式错误：keywords 必须包含 {MIN_KEYWORDS} 到 {MAX_KEYWORDS} 项"
            )
            keywords_are_valid = False
        if len(cleaned) != len(raw_keywords):
            issues.append("格式错误：keywords 只能包含非空字符串")
            keywords_are_valid = False
        normalized_keywords = [_normalize(keyword) for keyword in cleaned]
        if any(not keyword for keyword in normalized_keywords):
            issues.append("格式错误：keywords 不能只包含空白或标点")
            keywords_are_valid = False
        if len(set(normalized_keywords)) != len(normalized_keywords):
            issues.append("格式错误：keywords 不得重复")
            keywords_are_valid = False

    reference = _normalize(reference_answer)
    candidate = _normalize(candidate_answer)
    covered_keywords = tuple(
        keyword for keyword in keywords if _normalize(keyword) in candidate
    )
    coverage_ratio = len(covered_keywords) / len(keywords) if keywords else 0.0

    if keywords_are_valid:
        outside_reference = [
            keyword for keyword in keywords if _normalize(keyword) not in reference
        ]
        if outside_reference:
            issues.append(
                "关键词错误：以下关键词不在参考答案中："
                + ", ".join(outside_reference)
            )
            keywords_are_valid = False

    if score is not None and keywords_are_valid and keywords:
        conflict = _score_coverage_issue(score, len(covered_keywords), len(keywords))
        if conflict:
            issues.append(
                f"{conflict}，实际为 {coverage_ratio:.0%} "
                f"({len(covered_keywords)}/{len(keywords)})"
            )

    return ReviewReport(
        score=score,
        reason=reason,
        keywords=keywords,
        covered_keywords=covered_keywords,
        coverage_ratio=coverage_ratio,
        issues=tuple(issues),
    )
