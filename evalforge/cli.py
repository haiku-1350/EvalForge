from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import DeepSeekEvaluator, EvaluationError, EvaluationResult
from .rag import (
    DEFAULT_RAG_ENTRYPOINT,
    DEFAULT_RAG_PROJECT,
    PythonRagAdapter,
    RagAnswer,
    RagIntegrationError,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_cases.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "rag_evaluation_results.json"
CASE_FIELDS = {"question_id", "question", "reference_answer"}
CASE_COUNT = 3


@dataclass(frozen=True)
class RagCaseResult:
    rag_answer: RagAnswer
    evaluation: EvaluationResult


def load_test_cases(path: Path) -> list[dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"测试数据文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"测试数据不是合法 JSON：{path}") from exc
    if not isinstance(data, list) or len(data) != CASE_COUNT:
        raise EvaluationError(f"v2 测试数据必须恰好包含 {CASE_COUNT} 个问题")

    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise EvaluationError(f"第 {index} 个测试问题必须是 JSON 对象")
        if set(case) != CASE_FIELDS:
            missing = sorted(CASE_FIELDS - set(case))
            extra = sorted(set(case) - CASE_FIELDS)
            details: list[str] = []
            if missing:
                details.append("缺少 " + ", ".join(missing))
            if extra:
                details.append("包含多余字段 " + ", ".join(extra))
            raise EvaluationError(f"第 {index} 个测试问题字段不正确：{'；'.join(details)}")
        if not all(isinstance(case[field], str) and case[field].strip() for field in CASE_FIELDS):
            raise EvaluationError(f"第 {index} 个问题的文本字段不能为空")
        question_id = case["question_id"].strip()
        if question_id in seen_ids:
            raise EvaluationError(f"问题编码重复：{question_id}")
        seen_ids.add(question_id)
        normalized.append({field: case[field].strip() for field in CASE_FIELDS})
    return normalized


def evaluate_rag(
    evaluator: DeepSeekEvaluator,
    rag: PythonRagAdapter,
    cases: list[dict[str, str]],
) -> dict[str, RagCaseResult]:
    results: dict[str, RagCaseResult] = {}
    for case in cases:
        question_id = case["question_id"]
        print(f"\n[{question_id}] {case['question']}")
        rag_answer = rag.answer(case["question"])
        print(f"  RAG 回答：{rag_answer.text}")
        evaluation = evaluator.evaluate(
            case["question"],
            case["reference_answer"],
            rag_answer.text,
            question_id=question_id,
        )
        score_text = str(evaluation.score) if evaluation.score is not None else "无有效分数"
        review_text = "，需人工复核" if evaluation.needs_review else ""
        print(
            f"  评测：{score_text} 分{review_text}。"
            f"语义覆盖：{evaluation.semantic_coverage:.0%}，"
            f"词面重合：{evaluation.lexical_overlap:.0%}。"
            f"来源：{evaluation.stage}/{evaluation.model}。"
            f"理由：{evaluation.reason}"
        )
        if evaluation.validation_warnings:
            print(f"    警告：{'；'.join(evaluation.validation_warnings)}")
        if evaluation.needs_review:
            print(f"    复核原因：{'；'.join(evaluation.review_issues)}")
        results[question_id] = RagCaseResult(rag_answer, evaluation)
    return results


def check_rag_acceptance(
    cases: list[dict[str, str]],
    results: dict[str, RagCaseResult],
    min_score: int,
) -> list[str]:
    failures: list[str] = []
    for case in cases:
        question_id = case["question_id"]
        evaluation = results[question_id].evaluation
        if evaluation.needs_review or evaluation.score is None:
            failures.append(f"{question_id} 需要人工复核")
        elif evaluation.score < min_score:
            failures.append(
                f"{question_id} 的 RAG 回答为 {evaluation.score} 分，低于门槛 {min_score} 分"
            )
    return failures


def save_results(
    path: Path,
    cases: list[dict[str, str]],
    results: dict[str, RagCaseResult],
) -> None:
    payload = [
        _serialize_result(case, results[case["question_id"]]) for case in cases
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise EvaluationError(f"无法保存评测结果：{path}") from exc


def _serialize_result(
    case: dict[str, str], case_result: RagCaseResult
) -> dict[str, Any]:
    rag_answer = case_result.rag_answer
    result = case_result.evaluation
    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "reference_answer": case["reference_answer"],
        "rag_answer": rag_answer.text,
        "rag_call": {
            "project_root": rag_answer.project_root,
            "entrypoint": rag_answer.entrypoint,
            "duration_ms": rag_answer.duration_ms,
        },
        "final_score": result.score,
        "final_reason": result.reason,
        "semantic_coverage": result.semantic_coverage,
        "lexical_overlap": result.lexical_overlap,
        "judge_source": result.stage,
        "judge_model": result.model,
        "status": result.status,
        "attempt_count": result.attempts,
        "needs_review": result.needs_review,
        "candidate_self_contradiction": result.candidate_self_contradiction,
        "review_reasons": list(result.review_issues),
        "validation_warnings": list(result.validation_warnings),
        "key_point_judgments": [
            {
                "id": item.id,
                "status": item.status,
                "evidence": item.evidence,
                "explanation": item.explanation,
            }
            for item in result.key_point_judgments
        ],
        "extra_claims": [
            {
                "claim": item.claim,
                "status": item.status,
                "evidence": item.evidence,
                "severity": item.severity,
                "explanation": item.explanation,
            }
            for item in result.extra_claims
        ],
        "attempts": [
            {
                "stage": attempt.stage,
                "model": attempt.model,
                "passed": attempt.passed,
                "score": attempt.score,
                "duration_ms": attempt.duration_ms,
                "token_usage": attempt.token_usage,
                "raw_result": attempt.raw_result,
                "issues": list(attempt.issues),
                "warnings": list(attempt.warnings),
            }
            for attempt in result.review_history
        ],
        "reference_atomization_attempts": [
            {
                "stage": attempt.stage,
                "model": attempt.model,
                "passed": attempt.passed,
                "duration_ms": attempt.duration_ms,
                "token_usage": attempt.token_usage,
                "raw_result": attempt.raw_result,
                "issues": list(attempt.issues),
            }
            for attempt in result.reference_history
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvalForge v2 RAG 系统评测")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="三个问题及参考答案的 JSON 文件"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="结构化评测结果 JSON 路径",
    )
    parser.add_argument(
        "--rag-project",
        type=Path,
        default=Path(os.getenv("EVALFORGE_RAG_PROJECT", str(DEFAULT_RAG_PROJECT))),
        help="本地 RAG 项目根目录",
    )
    parser.add_argument(
        "--rag-entrypoint",
        default=os.getenv("EVALFORGE_RAG_ENTRYPOINT", DEFAULT_RAG_ENTRYPOINT),
        help="RAG Python 入口，格式为 module:function",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        choices=range(0, 6),
        default=4,
        help="RAG 回答自动验收的最低分，默认 4",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = load_test_cases(args.data)
        rag = PythonRagAdapter(args.rag_project, args.rag_entrypoint)
        evaluator = DeepSeekEvaluator()
        print(
            f"EvalForge v2 开始评测：RAG={rag.entrypoint}，"
            f"模型 A={evaluator.model_a}，模型 B={evaluator.model_b}，"
            f"共 {len(cases)} 个问题。"
        )
        results = evaluate_rag(evaluator, rag, cases)
        save_results(args.output, cases, results)
        print(f"\n结构化结果已保存：{args.output}")
        failures = check_rag_acceptance(cases, results, args.min_score)
    except (EvaluationError, RagIntegrationError) as exc:
        print(f"运行失败：{exc}")
        return 2
    except KeyboardInterrupt:
        print("\n运行已取消。")
        return 130

    if failures:
        print("\nRAG 验收未通过：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nRAG 验收通过：三个实时回答均达到自动验收门槛。")
    return 0
