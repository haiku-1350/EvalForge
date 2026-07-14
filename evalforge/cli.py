from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import (
    DeepSeekEvaluator,
    EvaluationError,
    EvaluationResult,
    GroundednessEvaluationResult,
)
from .rag import (
    DEFAULT_RAG_ENTRYPOINT,
    DEFAULT_RAG_PROJECT,
    HttpRagAdapter,
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
    correctness: EvaluationResult
    retrieval: EvaluationResult
    groundedness: GroundednessEvaluationResult
    error_type: str


def load_test_cases(path: Path) -> list[dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"测试数据文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"测试数据不是合法 JSON：{path}") from exc
    if not isinstance(data, list) or len(data) != CASE_COUNT:
        raise EvaluationError(f"v2.2 测试数据必须恰好包含 {CASE_COUNT} 个问题")

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
    rag: PythonRagAdapter | HttpRagAdapter,
    cases: list[dict[str, str]],
    threshold: int = 4,
) -> dict[str, RagCaseResult]:
    results: dict[str, RagCaseResult] = {}
    for case in cases:
        question_id = case["question_id"]
        print(f"\n[{question_id}] {case['question']}")
        rag_answer = rag.answer(case["question"], question_id=question_id)
        print(f"  RAG 回答：{rag_answer.text}")
        if not rag_answer.trace_available:
            raise RagIntegrationError(
                "v2.2 需要 RAG 返回检索轨迹；HTTP 响应必须包含 retrieved_context"
            )
        print(
            f"  检索：intent={rag_answer.intent}，改写={rag_answer.rewritten_question}，"
            f"内容={rag_answer.retrieved_context or '（空）'}"
        )
        correctness = evaluator.evaluate(
            case["question"],
            case["reference_answer"],
            rag_answer.text,
            question_id=question_id,
        )
        retrieval = evaluator.evaluate(
            case["question"],
            case["reference_answer"],
            rag_answer.retrieved_context,
            question_id=question_id,
        )
        groundedness = evaluator.evaluate_groundedness(
            case["question"],
            rag_answer.retrieved_context,
            rag_answer.text,
            question_id=question_id,
        )
        error_type = classify_error_type(
            correctness, retrieval, groundedness, threshold
        )
        correctness_text = (
            str(correctness.score) if correctness.score is not None else "无有效分数"
        )
        groundedness_text = (
            str(groundedness.score)
            if groundedness.score is not None
            else "无有效分数"
        )
        retrieval_text = (
            str(retrieval.score) if retrieval.score is not None else "无有效分数"
        )
        needs_review = (
            correctness.needs_review
            or retrieval.needs_review
            or groundedness.needs_review
        )
        review_text = "，需人工复核" if needs_review else ""
        print(
            f"  评测：correctness={correctness_text}，"
            f"groundedness={groundedness_text}，retrieval={retrieval_text}，"
            f"error_type={error_type}{review_text}。"
        )
        print(f"    正确性理由：{correctness.reason}")
        print(f"    忠实度理由：{groundedness.reason}")
        results[question_id] = RagCaseResult(
            rag_answer,
            correctness,
            retrieval,
            groundedness,
            error_type,
        )
    return results


def classify_error_type(
    correctness: EvaluationResult,
    retrieval: EvaluationResult,
    groundedness: GroundednessEvaluationResult,
    threshold: int,
) -> str:
    correctness_bad = _evaluation_failed(correctness, threshold)
    retrieval_bad = _evaluation_failed(retrieval, threshold)
    groundedness_bad = (
        groundedness.needs_review
        or groundedness.score is None
        or groundedness.score < threshold
    )
    if retrieval_bad and groundedness_bad:
        return "both"
    if retrieval_bad:
        return "retrieval"
    if groundedness_bad or correctness_bad:
        return "generation"
    return "none"


def _evaluation_failed(result: EvaluationResult, threshold: int) -> bool:
    return result.needs_review or result.score is None or result.score < threshold


def check_rag_acceptance(
    cases: list[dict[str, str]],
    results: dict[str, RagCaseResult],
    min_score: int,
) -> list[str]:
    failures: list[str] = []
    for case in cases:
        question_id = case["question_id"]
        case_result = results[question_id]
        if (
            case_result.correctness.needs_review
            or case_result.retrieval.needs_review
            or case_result.groundedness.needs_review
        ):
            failures.append(f"{question_id} 需要人工复核")
        elif case_result.error_type != "none":
            failures.append(
                f"{question_id} error_type={case_result.error_type}："
                f"correctness={case_result.correctness.score}，"
                f"groundedness={case_result.groundedness.score}，"
                f"retrieval={case_result.retrieval.score}，门槛={min_score}"
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
    result = case_result.correctness
    retrieval = case_result.retrieval
    groundedness = case_result.groundedness
    needs_review = (
        result.needs_review or retrieval.needs_review or groundedness.needs_review
    )
    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "reference_answer": case["reference_answer"],
        "rag_answer": rag_answer.text,
        "rag_call": {
            "project_root": rag_answer.project_root,
            "entrypoint": rag_answer.entrypoint,
            "transport": rag_answer.transport,
            "duration_ms": rag_answer.duration_ms,
            "intent": rag_answer.intent,
            "rewritten_question": rag_answer.rewritten_question,
            "retrieved_context": rag_answer.retrieved_context,
            "need_human": rag_answer.need_human,
        },
        "correctness_score": result.score,
        "groundedness_score": groundedness.score,
        "error_type": case_result.error_type,
        "retrieval_score": retrieval.score,
        "final_score": result.score,
        "final_reason": result.reason,
        "semantic_coverage": result.semantic_coverage,
        "lexical_overlap": result.lexical_overlap,
        "judge_source": result.stage,
        "judge_model": result.model,
        "status": result.status,
        "attempt_count": result.attempts,
        "needs_review": needs_review,
        "candidate_self_contradiction": result.candidate_self_contradiction,
        "review_reasons": [
            *(f"correctness: {issue}" for issue in result.review_issues),
            *(f"retrieval: {issue}" for issue in retrieval.review_issues),
            *(f"groundedness: {issue}" for issue in groundedness.review_issues),
        ],
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
        "retrieval_evaluation": {
            "score": retrieval.score,
            "reason": retrieval.reason,
            "status": retrieval.status,
            "needs_review": retrieval.needs_review,
            "attempts": [_serialize_attempt(attempt) for attempt in retrieval.review_history],
        },
        "groundedness_evaluation": {
            "score": groundedness.score,
            "reason": groundedness.reason,
            "status": groundedness.status,
            "needs_review": groundedness.needs_review,
            "claims": [
                {
                    "claim": claim.claim,
                    "status": claim.status,
                    "answer_evidence": claim.answer_evidence,
                    "context_evidence": claim.context_evidence,
                    "explanation": claim.explanation,
                }
                for claim in groundedness.claims
            ],
            "attempts": [
                _serialize_attempt(attempt)
                for attempt in groundedness.review_history
            ],
        },
    }


def _serialize_attempt(attempt: Any) -> dict[str, Any]:
    return {
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


def _http_timeout(value: str) -> int:
    try:
        timeout = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("超时必须是整数") from exc
    if not 1 <= timeout <= 300:
        raise argparse.ArgumentTypeError("超时必须在 1 到 300 秒之间")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvalForge v2.2 RAG 归因评测")
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
        "--rag-transport",
        choices=("python", "http"),
        default=os.getenv("EVALFORGE_RAG_TRANSPORT", "python"),
        help="RAG 接入方式，默认 python",
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
        "--rag-url",
        default=os.getenv("EVALFORGE_RAG_URL", ""),
        help="外部 RAG HTTP 接口地址",
    )
    parser.add_argument(
        "--rag-timeout",
        type=_http_timeout,
        default=os.getenv("EVALFORGE_RAG_TIMEOUT", "60"),
        help="外部 RAG HTTP 超时秒数，默认 60",
    )
    parser.add_argument(
        "--rag-api-key-env",
        default=os.getenv("EVALFORGE_RAG_API_KEY_ENV", "EVALFORGE_RAG_API_KEY"),
        help="保存外部 RAG Bearer Token 的环境变量名称",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        choices=range(0, 6),
        default=4,
        help="RAG 回答自动验收的最低分，默认 4",
    )
    return parser


def build_rag_adapter(args: argparse.Namespace) -> PythonRagAdapter | HttpRagAdapter:
    if args.rag_transport == "http":
        if not args.rag_url:
            raise RagIntegrationError(
                "HTTP 接入需要 --rag-url 或环境变量 EVALFORGE_RAG_URL"
            )
        api_key = os.getenv(args.rag_api_key_env, "") if args.rag_api_key_env else ""
        return HttpRagAdapter(
            args.rag_url,
            api_key=api_key,
            timeout_seconds=args.rag_timeout,
        )
    return PythonRagAdapter(args.rag_project, args.rag_entrypoint)


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = load_test_cases(args.data)
        rag = build_rag_adapter(args)
        evaluator = DeepSeekEvaluator()
        print(
            f"EvalForge v2.2 开始评测：RAG={rag.entrypoint}，"
            f"模型 A={evaluator.model_a}，模型 B={evaluator.model_b}，"
            f"共 {len(cases)} 个问题。"
        )
        results = evaluate_rag(evaluator, rag, cases, args.min_score)
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
