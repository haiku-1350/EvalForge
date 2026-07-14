from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluator import DeepSeekEvaluator, EvaluationError, EvaluationResult, MODEL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_cases.json"
ANSWER_TYPE_LABELS = {
    "correct": "完全正确",
    "partial": "部分正确",
    "incorrect": "完全错误",
}
REQUIRED_TYPES = tuple(ANSWER_TYPE_LABELS)


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
            if answer_type not in REQUIRED_TYPES:
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
                case["question"], case["reference_answer"], answer["answer"]
            )
            case_results[answer_type] = result
            print(
                f"  {ANSWER_TYPE_LABELS[answer_type]}答案：{result.score} 分。"
                f"评分理由：{result.reason}"
            )
        results[question_id] = case_results
    return results


def check_score_acceptance(
    cases: list[dict[str, Any]], results: dict[str, dict[str, EvaluationResult]]
) -> list[str]:
    failures: list[str] = []
    for case in cases:
        question_id = case["question_id"]
        scores = results[question_id]
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
        scores = [
            evaluator.evaluate(case["question"], case["reference_answer"], answer).score
            for _ in range(3)
        ]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvalForge V1 LLM 答案评测")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="测试数据 JSON 路径"
    )
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="评测全部数据，并额外执行三类答案的稳定性验收",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = load_test_cases(args.data)
        evaluator = DeepSeekEvaluator()
        print(f"EvalForge V1 开始评测：模型固定为 {MODEL}，共 {len(cases)} 个问题。")
        results = evaluate_all(evaluator, cases)
        failures = check_score_acceptance(cases, results)
        if args.acceptance:
            failures.extend(run_stability_check(evaluator, cases))
    except EvaluationError as exc:
        print(f"运行失败：{exc}")
        return 1
    except KeyboardInterrupt:
        print("\n运行已取消。")
        return 130

    if failures:
        print("\n验收未通过：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n验收通过：JSON 格式、分数边界和三类答案排序均符合要求。")
    return 0
