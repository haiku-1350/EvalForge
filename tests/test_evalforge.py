import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evalforge.cli import (
    RagCaseResult,
    check_rag_acceptance,
    load_test_cases,
    save_results,
)
from evalforge.evaluator import DeepSeekEvaluator, EvaluationError, EvaluationResult
from evalforge.rag import PythonRagAdapter, RagAnswer, RagIntegrationError
from evalforge.reviewer import (
    KeyPoint,
    KeyPointJudgment,
    find_judge_disagreement,
    review_reference_content,
    review_result,
)


QUESTION_ID = "Q001"
QUESTION = "列表最重要的特性是什么？"
REFERENCE = "列表创建后可以修改元素；列表通常使用方括号表示。"
CANDIDATE = "列表可以修改。"
KEY_POINTS = (
    KeyPoint("K1", "列表创建后可以修改元素", "core"),
    KeyPoint("K2", "列表通常使用方括号表示", "supporting"),
)
MODEL_B_ENV = {
    "GLM_API_KEY": "glm-key",
}


def api_response(result: object) -> dict:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def reference_payload(*, defect: bool = False) -> dict:
    return {
        "question_id": QUESTION_ID,
        "reference_defect": defect,
        "reference_defect_reasons": ["参考答案自相矛盾"] if defect else [],
        "key_points": []
        if defect
        else [
            {
                "id": point.id,
                "statement": point.statement,
                "importance": point.importance,
            }
            for point in KEY_POINTS
        ],
    }


def judge_payload(
    *,
    score: int = 4,
    core_status: str = "matched",
    supporting_status: str = "missing",
    uncertain: bool = False,
    self_contradiction: bool = False,
) -> dict:
    core_evidence = "列表可以修改" if core_status != "missing" else None
    supporting_evidence = "列表可以修改" if supporting_status != "missing" else None
    return {
        "question_id": QUESTION_ID,
        "key_point_judgments": [
            {
                "id": "K1",
                "status": core_status,
                "evidence": core_evidence,
                "explanation": "语义等价" if core_status == "matched" else "判断说明",
            },
            {
                "id": "K2",
                "status": supporting_status,
                "evidence": supporting_evidence,
                "explanation": "没有说明方括号"
                if supporting_status == "missing"
                else "判断说明",
            },
        ],
        "extra_claims": [],
        "candidate_self_contradiction": self_contradiction,
        "score": score,
        "reason": "核心事实正确，仅遗漏补充信息。",
        "uncertain": uncertain,
        "uncertainty_reasons": ["答案内部立场矛盾"] if uncertain else [],
    }


class ReferenceReviewerTests(unittest.TestCase):
    def test_accepts_valid_atomic_key_points(self) -> None:
        report = review_reference_content(
            json.dumps(reference_payload(), ensure_ascii=False), QUESTION_ID
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.analysis.key_points, KEY_POINTS)

    def test_requires_at_least_one_core_point(self) -> None:
        payload = reference_payload()
        payload["key_points"][0]["importance"] = "supporting"
        report = review_reference_content(json.dumps(payload), QUESTION_ID)
        self.assertFalse(report.passed)
        self.assertTrue(any("至少需要一个 core" in issue for issue in report.issues))

    def test_reference_defect_requires_reason(self) -> None:
        payload = reference_payload(defect=True)
        payload["reference_defect_reasons"] = []
        report = review_reference_content(json.dumps(payload), QUESTION_ID)
        self.assertFalse(report.passed)
        self.assertTrue(any("reference_defect" in issue for issue in report.issues))


class SemanticReviewerTests(unittest.TestCase):
    def test_accepts_score_four_with_missing_supporting_point(self) -> None:
        report = review_result(
            judge_payload(), QUESTION_ID, REFERENCE, CANDIDATE, KEY_POINTS
        )
        self.assertTrue(report.acceptable)
        self.assertAlmostEqual(report.semantic_coverage, 2 / 3)

    def test_score_five_conflicts_with_missing_point(self) -> None:
        report = review_result(
            judge_payload(score=5), QUESTION_ID, REFERENCE, CANDIDATE, KEY_POINTS
        )
        self.assertFalse(report.passed)
        self.assertIn("score_state_conflict", report.issue_codes)

    def test_rejects_evidence_not_found_in_candidate(self) -> None:
        payload = judge_payload()
        payload["key_point_judgments"][0]["evidence"] = "不存在的原文"
        report = review_result(payload, QUESTION_ID, REFERENCE, CANDIDATE, KEY_POINTS)
        self.assertIn("evidence_not_found", report.issue_codes)

    def test_missing_status_requires_null_evidence(self) -> None:
        payload = judge_payload()
        payload["key_point_judgments"][1]["evidence"] = "列表"
        report = review_result(payload, QUESTION_ID, REFERENCE, CANDIDATE, KEY_POINTS)
        self.assertIn("schema_invalid", report.issue_codes)

    def test_low_lexical_overlap_is_warning_only(self) -> None:
        points = (KeyPoint("K1", "列表创建后可以修改元素", "core"),)
        candidate = "这个序列事后允许变更内容。"
        payload = {
            "question_id": QUESTION_ID,
            "key_point_judgments": [
                {
                    "id": "K1",
                    "status": "matched",
                    "evidence": "允许变更内容",
                    "explanation": "语义等价",
                }
            ],
            "extra_claims": [],
            "candidate_self_contradiction": False,
            "score": 5,
            "reason": "语义完整。",
            "uncertain": False,
            "uncertainty_reasons": [],
        }
        report = review_result(payload, QUESTION_ID, REFERENCE, candidate, points)
        self.assertTrue(report.acceptable)
        self.assertLess(report.lexical_overlap, 0.5)
        self.assertEqual(report.warnings, ("lexical_semantic_mismatch",))

    def test_uncertain_result_is_valid_but_not_acceptable(self) -> None:
        report = review_result(
            judge_payload(uncertain=True),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
            KEY_POINTS,
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.acceptable)

    def test_self_contradiction_requires_uncertainty(self) -> None:
        report = review_result(
            judge_payload(self_contradiction=True),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
            KEY_POINTS,
        )
        self.assertIn("self_contradiction_conflict", report.issue_codes)

    def test_self_contradiction_with_uncertainty_goes_to_review(self) -> None:
        report = review_result(
            judge_payload(uncertain=True, self_contradiction=True),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
            KEY_POINTS,
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.acceptable)
        self.assertTrue(report.candidate_self_contradiction)

    def test_detects_large_judge_score_disagreement(self) -> None:
        model_a = review_result(
            judge_payload(score=4, uncertain=True),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
            KEY_POINTS,
        )
        model_b = review_result(
            judge_payload(score=2),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
            KEY_POINTS,
        )
        reasons = find_judge_disagreement(model_a, model_b, KEY_POINTS)
        self.assertTrue(any("相差 2 分" in reason for reason in reasons))


class EvaluatorTests(unittest.TestCase):
    def test_requires_model_a_environment_variable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EvaluationError, "DEEPSEEK_API_KEY"):
                DeepSeekEvaluator()

    def test_accepts_model_a_initial_result_and_records_audit_data(self) -> None:
        responses = [reference_payload(), judge_payload()]

        def fake_post(url, headers, body, timeout):
            return api_response(responses.pop(0))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertEqual(result.status, "accepted_model_a")
        self.assertEqual(result.score, 4)
        self.assertEqual(result.review_history[0].token_usage["total_tokens"], 30)
        self.assertTrue(result.review_history[0].raw_result)
        self.assertEqual(len(result.reference_history), 1)

    def test_reference_atomization_is_cached_for_all_answers(self) -> None:
        responses = [reference_payload(), judge_payload(), judge_payload()]
        request_bodies = []

        def fake_post(url, headers, body, timeout):
            request_bodies.append(json.loads(body))
            return api_response(responses.pop(0))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            evaluator = DeepSeekEvaluator(http_post=fake_post)
            evaluator.evaluate(QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID)
            evaluator.evaluate(QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID)

        atomization_calls = [
            body
            for body in request_bodies
            if "请原子化" in body["messages"][1]["content"]
        ]
        self.assertEqual(len(atomization_calls), 1)

    def test_model_a_receives_hard_failure_and_corrects_once(self) -> None:
        responses = [reference_payload(), judge_payload(score=5), judge_payload(score=4)]
        request_bodies = []

        def fake_post(url, headers, body, timeout):
            request_bodies.append(json.loads(body))
            return api_response(responses.pop(0))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertEqual(result.status, "accepted_model_a_retry")
        self.assertEqual(result.attempts, 2)
        correction_prompt = request_bodies[2]["messages"][1]["content"]
        self.assertIn("score_state_conflict", correction_prompt)
        self.assertNotIn("词面", correction_prompt)

    def test_model_b_blind_result_is_accepted(self) -> None:
        responses = [
            reference_payload(),
            judge_payload(score=5),
            judge_payload(score=5),
            judge_payload(score=4),
        ]
        requests = []

        def fake_post(url, headers, body, timeout):
            requests.append((url, headers, json.loads(body)))
            return api_response(responses.pop(0))

        environment = {"DEEPSEEK_API_KEY": "a-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertEqual(result.status, "accepted_model_b")
        self.assertFalse(result.needs_review)
        self.assertEqual(result.model, "glm-5.1")
        self.assertEqual(
            requests[3][0],
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        )
        blind_prompt = requests[3][2]["messages"][1]["content"]
        self.assertNotIn("上一次", blind_prompt)
        self.assertNotIn("模型 A", blind_prompt)
        self.assertNotIn("partial", blind_prompt)

    def test_model_b_uncertainty_requires_human_review(self) -> None:
        responses = [
            reference_payload(),
            judge_payload(score=5),
            judge_payload(score=5),
            judge_payload(score=4, uncertain=True),
        ]

        def fake_post(url, headers, body, timeout):
            return api_response(responses.pop(0))

        environment = {"DEEPSEEK_API_KEY": "a-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertTrue(result.needs_review)
        self.assertEqual(result.status, "needs_human_review")
        self.assertTrue(any("judge_uncertain" in issue for issue in result.review_issues))

    def test_valid_b_result_with_large_disagreement_requires_review(self) -> None:
        responses = [
            reference_payload(),
            judge_payload(score=4, uncertain=True),
            judge_payload(score=4, uncertain=True),
            judge_payload(score=2),
        ]

        def fake_post(url, headers, body, timeout):
            return api_response(responses.pop(0))

        environment = {"DEEPSEEK_API_KEY": "a-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertTrue(result.needs_review)
        self.assertTrue(any("judge_disagreement" in issue for issue in result.review_issues))

    def test_reference_defect_goes_directly_to_human(self) -> None:
        def fake_post(url, headers, body, timeout):
            return api_response(reference_payload(defect=True))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
            )

        self.assertTrue(result.needs_review)
        self.assertEqual(result.status, "reference_defect")
        self.assertEqual(result.attempts, 0)

    def test_model_b_key_is_required_only_when_blind_review_runs(self) -> None:
        responses = [reference_payload(), judge_payload(score=5), judge_payload(score=5)]

        def fake_post(url, headers, body, timeout):
            return api_response(responses.pop(0))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            evaluator = DeepSeekEvaluator(http_post=fake_post)
            with self.assertRaisesRegex(EvaluationError, "GLM_API_KEY"):
                evaluator.evaluate(
                    QUESTION, REFERENCE, CANDIDATE, question_id=QUESTION_ID
                )


class CliTests(unittest.TestCase):
    def test_bundled_data_contains_exactly_three_rag_questions(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "test_cases.json"
        cases = load_test_cases(path)
        self.assertEqual(len(cases), 3)
        self.assertEqual(
            set(cases[0]), {"question_id", "question", "reference_answer"}
        )

    def test_rejects_offline_candidate_answers(self) -> None:
        cases = [
            {
                "question_id": f"Q00{index}",
                "question": "问题",
                "reference_answer": "参考答案",
                "answers": [],
            }
            for index in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "多余字段 answers"):
                load_test_cases(path)

    def test_saves_structured_audit_result(self) -> None:
        result = EvaluationResult(
            score=4,
            reason="核心正确。",
            key_point_judgments=(
                KeyPointJudgment("K1", "matched", "列表可以修改", "语义等价"),
            ),
            semantic_coverage=1.0,
            lexical_overlap=0.2,
            model="glm-5.1",
            stage="model_b_blind",
            status="accepted_model_b",
        )
        cases = [{
            "question_id": QUESTION_ID,
            "question": QUESTION,
            "reference_answer": REFERENCE,
        }]
        case_result = RagCaseResult(
            rag_answer=RagAnswer(
                text=CANDIDATE,
                duration_ms=123,
                project_root=r"E:\Enterprise AI Helpdesk",
                entrypoint="utils.answer:answer_user",
            ),
            evaluation=result,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_results(path, cases, {QUESTION_ID: case_result})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload[0]["rag_answer"], CANDIDATE)
        self.assertEqual(payload[0]["rag_call"]["duration_ms"], 123)
        self.assertEqual(payload[0]["judge_source"], "model_b_blind")
        self.assertEqual(payload[0]["semantic_coverage"], 1.0)
        self.assertEqual(payload[0]["key_point_judgments"][0]["id"], "K1")

    def test_rag_acceptance_uses_score_threshold(self) -> None:
        cases = [
            {"question_id": QUESTION_ID, "question": QUESTION, "reference_answer": REFERENCE}
        ]
        case_result = RagCaseResult(
            RagAnswer(CANDIDATE, 1, "demo", "demo:answer"),
            EvaluationResult(score=3, reason="不完整"),
        )
        failures = check_rag_acceptance(cases, {QUESTION_ID: case_result}, 4)
        self.assertEqual(len(failures), 1)
        self.assertIn("低于门槛 4 分", failures[0])


class RagAdapterTests(unittest.TestCase):
    def test_calls_configured_python_entrypoint_from_project_directory(self) -> None:
        module_name = "evalforge_test_rag_api"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{module_name}.py").write_text(
                "from pathlib import Path\n"
                "def answer(question):\n"
                "    return f'{Path.cwd().name}:{question}'\n",
                encoding="utf-8",
            )
            try:
                adapter = PythonRagAdapter(root, f"{module_name}:answer")
                result = adapter.answer("测试问题")
                cache_created = (root / "__pycache__").exists()
            finally:
                import sys

                sys.modules.pop(module_name, None)
        self.assertEqual(result.text, f"{root.name}:测试问题")
        self.assertEqual(result.entrypoint, f"{module_name}:answer")
        self.assertFalse(cache_created)

    def test_rejects_empty_rag_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = PythonRagAdapter(Path(directory), "demo:answer", lambda _: "")
            with self.assertRaisesRegex(RagIntegrationError, "空答案"):
                adapter.answer("测试问题")


if __name__ == "__main__":
    unittest.main()
