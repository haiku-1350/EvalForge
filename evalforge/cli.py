from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluator import DeepSeekEvaluator, EvaluationError, EvaluationResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_cases.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "evaluation_results.json"
ANSWER_TYPE_LABELS = {
    "correct": "完全正确",
    "partial": "部分正确",
    "incorrect": "完全错误",
    "ambiguous": "含糊矛盾",
}
REQUIRED_TYPES = ("correct", "partial", "incorrect")


def load_test_cases(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"测试数据文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"测试数据不是合法 JSON：{path}") from exc
    if not isinstance(data, list) or len(data) < 3:
        raise EvaluationError("测试数据必须是至少包含 3 个问题的 JSON 数组")

    seen_ids: set[str] = set()
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise EvaluationError(f"第 {index} 个测试问题必须是 JSON 对象")
        for field in ("question_id", "question", "reference_answer", "answers"):
            if field not in case:
                raise EvaluationError(f"第 {index} 个测试问题缺少字段 {field}")
        if not all(
            isinstance(case[field], str) and case[field].strip()
            for field in ("question_id", "question", "reference_answer")
        ):
            raise EvaluationError(f"第 {index} 个问题的文本字段不能为空")
        if case["question_id"] in seen_ids:
            raise EvaluationError(f"问题编码重复：{case['question_id']}")
        seen_ids.add(case["question_id"])

        answers = case["answers"]
        if not isinstance(answers, list):
            raise EvaluationError(f"问题 {case['question_id']} 的 answers 必须是数组")
        by_type: dict[str, int] = defaultdict(int)
        for answer in answers:
            if not isinstance(answer, dict):
                raise EvaluationError(f"问题 {case['question_id']} 包含非法答案")
            answer_type = answer.get("answer_type")
            text = answer.get("answer")
            if answer_type not in ANSWER_TYPE_LABELS:
                raise EvaluationError(
                    f"问题 {case['question_id']} 包含未知 answer_type：{answer_type}"
                )
            if not isinstance(text, str) or not text.strip():
                raise EvaluationError(f"问题 {case['question_id']} 包含空答案")
            by_type[answer_type] += 1
        missing = [kind for kind in REQUIRED_TYPES if by_type[kind] == 0]
        if missing:
            raise EvaluationError(
                f"问题 {case['question_id']} 缺少答案类型：{', '.join(missing)}"
            )
    return data


def evaluate_all(
    evaluator: DeepSeekEvaluator, cases: list[dict[str, Any]]
) -> dict[str, dict[str, EvaluationResult]]:
    results: dict[str, dict[str, EvaluationResult]] = {}
    for case in cases:
        question_id = case["question_id"]
        print(f"\n[{question_id}] {case['question']}")
        case_results: dict[str, EvaluationResult] = {}
        for answer in case["answers"]:
            answer_type = answer["answer_type"]
            result = evaluator.evaluate(
                case["question"],
                case["reference_answer"],
                answer["answer"],
                question_id=question_id,
            )
            case_results[answer_type] = result
            score_text = str(result.score) if result.score is not None else "无有效分数"
            review_text = "，需人工复核" if result.needs_review else ""
            print(
                f"  {ANSWER_TYPE_LABELS[answer_type]}答案：{score_text} 分{review_text}。"
                f"语义覆盖：{result.semantic_coverage:.0%}，"
                f"词面重合：{result.lexical_overlap:.0%}。"
                f"判定来源：{result.stage}/{result.model}。评分理由：{result.reason}"
            )
            if result.validation_warnings:
                print(f"    警告：{'；'.join(result.validation_warnings)}")
            if result.needs_review:
                print(f"    复核原因：{'；'.join(result.review_issues)}")
        results[question_id] = case_results
    return results


def check_score_acceptance(
    cases: list[dict[str, Any]], results: dict[str, dict[str, EvaluationResult]]
) -> list[str]:
    failures: list[str] = []
    for case in cases:
        question_id = case["question_id"]
        scores = results[question_id]
        required_review = [
            answer_type
            for answer_type in REQUIRED_TYPES
            if scores[answer_type].needs_review or scores[answer_type].score is None
        ]
        if required_review:
            failures.append(
                f"{question_id} 存在需人工复核的基础答案：{', '.join(required_review)}"
            )
            continue
        if scores["correct"].score < 4:
            failures.append(f"{question_id} 的完全正确答案低于 4 分")
        if scores["incorrect"].score > 1:
            failures.append(f"{question_id} 的完全错误答案高于 1 分")
        if not (
            scores["correct"].score
            > scores["partial"].score
            > scores["incorrect"].score
        ):
            failures.append(f"{question_id} 的三类答案分数未严格递减")
        if "ambiguous" in scores:
            if not scores["ambiguous"].needs_review:
                failures.append(f"{question_id} 的含糊矛盾答案未进入人工复核")
            else:
                failures.append(f"{question_id} 存在需人工复核的含糊矛盾答案")
    return failures


def run_stability_check(
    evaluator: DeepSeekEvaluator, cases: list[dict[str, Any]]
) -> list[str]:
    selections = zip(cases[:3], REQUIRED_TYPES, strict=True)
    failures: list[str] = []
    print("\n稳定性测试（每条固定样本运行 3 次）：")
    for case, answer_type in selections:
        answer = next(
            item["answer"]
            for item in case["answers"]
            if item["answer_type"] == answer_type
        )
        evaluations = [
            evaluator.evaluate(
                case["question"],
                case["reference_answer"],
                answer,
                question_id=case["question_id"],
            )
            for _ in range(3)
        ]
        if any(result.needs_review or result.score is None for result in evaluations):
            failures.append(
                f"{case['question_id']} 的{ANSWER_TYPE_LABELS[answer_type]}样本需人工复核"
            )
            continue
        scores = [result.score for result in evaluations]
        spread = max(scores) - min(scores)
        print(
            f"  {case['question_id']} / {ANSWER_TYPE_LABELS[answer_type]}："
            f"{scores}，极差 {spread}"
        )
        if spread > 1:
            failures.append(
                f"{case['question_id']} 的{ANSWER_TYPE_LABELS[answer_type]}样本分差大于 1"
            )
    return failures


def save_results(
    path: Path,
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, EvaluationResult]],
) -> None:
    payload: list[dict[str, Any]] = []
    for case in cases:
        for answer in case["answers"]:
            result = results[case["question_id"]][answer["answer_type"]]
            payload.append(
                _serialize_result(
                    case["question_id"],
                    case["question"],
                    case["reference_answer"],
                    answer["answer"],
                    answer["answer_type"],
                    result,
                )
            )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise EvaluationError(f"无法保存评测结果：{path}") from exc


def _serialize_result(
    question_id: str,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    answer_type: str,
    result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "reference_answer": reference_answer,
        "candidate_answer": candidate_answer,
        "offline_answer_type": answer_type,
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
    parser = argparse.ArgumentParser(
        description="EvalForge v1.3 原子知识点语义评测"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="测试数据 JSON 路径"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="结构化评测结果 JSON 路径",
    )
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="评测全部数据，并额外执行稳定性验收",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = load_test_cases(args.data)
        evaluator = DeepSeekEvaluator()
        print(
            f"EvalForge v1.3 开始评测：模型 A={evaluator.model_a}，"
            f"模型 B={evaluator.model_b}，共 {len(cases)} 个问题。"
        )
        results = evaluate_all(evaluator, cases)
        save_results(args.output, cases, results)
        print(f"\n结构化结果已保存：{args.output}")
        failures = check_score_acceptance(cases, results)
        if args.acceptance:
            failures.extend(run_stability_check(evaluator, cases))
    except EvaluationError as exc:
        print(f"运行失败：{exc}")
        return 2
    except KeyboardInterrupt:
        print("\n运行已取消。")
        return 130

    if failures:
        print("\n验收未通过：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n验收通过：语义状态、证据、分数逻辑和双模型流程均符合要求。")
    return 0
