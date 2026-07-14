import argparse
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from evalforge.cli import (
    RagCaseResult,
    build_rag_adapter,
    check_rag_acceptance,
    classify_error_type,
    load_test_cases,
    save_results,
)
from evalforge.evaluator import (
    DeepSeekEvaluator,
    EvaluationError,
    EvaluationResult,
    GroundednessEvaluationResult,
)
from evalforge.grounding import GroundingClaim, review_grounding_result
from evalforge.rag import (
    HttpRagAdapter,
    PythonRagAdapter,
    RagAnswer,
    RagIntegrationError,
)
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


def grounding_payload(
    *, score: int = 5, status: str = "supported", uncertain: bool = False
) -> dict:
    context_evidence = "列表创建后可以修改元素" if status in {
        "supported",
        "partially_supported",
    } else None
    return {
        "question_id": QUESTION_ID,
        "claims": [
            {
                "claim": "列表可以修改",
                "status": status,
                "answer_evidence": "列表可以修改",
                "context_evidence": context_evidence,
                "explanation": "检索内容支持该陈述",
            }
        ],
        "groundedness_score": score,
        "reason": "回答中的事实有检索依据。",
        "uncertain": uncertain,
        "uncertainty_reasons": ["支持关系不清晰"] if uncertain else [],
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

    def test_accepts_groundedness_result_from_model_a(self) -> None:
        def fake_post(url, headers, body, timeout):
            return api_response(grounding_payload())

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "a-key"}, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate_groundedness(
                QUESTION,
                REFERENCE,
                CANDIDATE,
                question_id=QUESTION_ID,
            )

        self.assertEqual(result.score, 5)
        self.assertEqual(result.status, "accepted_model_a")
        self.assertEqual(result.claims[0].status, "supported")

    def test_groundedness_uses_model_b_after_failed_correction(self) -> None:
        responses = [
            grounding_payload(score=5, status="unsupported"),
            grounding_payload(score=5, status="unsupported"),
            grounding_payload(),
        ]
        requests = []

        def fake_post(url, headers, body, timeout):
            requests.append(json.loads(body))
            return api_response(responses.pop(0))

        environment = {"DEEPSEEK_API_KEY": "a-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate_groundedness(
                QUESTION,
                REFERENCE,
                CANDIDATE,
                question_id=QUESTION_ID,
            )

        self.assertEqual(result.status, "accepted_model_b")
        self.assertEqual(result.model, "glm-5.1")
        blind_prompt = requests[2]["messages"][1]["content"]
        self.assertNotIn("上一次", blind_prompt)


class GroundingReviewerTests(unittest.TestCase):
    def test_accepts_supported_claim_with_exact_evidence(self) -> None:
        report = review_grounding_result(
            grounding_payload(), QUESTION_ID, REFERENCE, CANDIDATE
        )
        self.assertTrue(report.acceptable)
        self.assertEqual(report.score, 5)

    def test_rejects_context_evidence_not_in_retrieval(self) -> None:
        payload = grounding_payload()
        payload["claims"][0]["context_evidence"] = "不存在的检索原文"
        report = review_grounding_result(
            payload, QUESTION_ID, REFERENCE, CANDIDATE
        )
        self.assertTrue(any("context_evidence" in issue for issue in report.issues))

    def test_score_five_rejects_unsupported_claim(self) -> None:
        report = review_grounding_result(
            grounding_payload(score=5, status="unsupported"),
            QUESTION_ID,
            REFERENCE,
            CANDIDATE,
        )
        self.assertTrue(any("score_state_conflict" in issue for issue in report.issues))

    def test_empty_context_allows_honest_non_factual_abstention(self) -> None:
        answer = "目前没有相关信息。"
        payload = {
            "question_id": QUESTION_ID,
            "claims": [
                {
                    "claim": "没有相关信息",
                    "status": "non_factual",
                    "answer_evidence": answer,
                    "context_evidence": None,
                    "explanation": "这是对空检索结果的诚实说明",
                }
            ],
            "groundedness_score": 5,
            "reason": "没有编造事实。",
            "uncertain": False,
            "uncertainty_reasons": [],
        }
        report = review_grounding_result(payload, QUESTION_ID, "", answer)
        self.assertTrue(report.acceptable)


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
        cases = [
            {
                "question_id": QUESTION_ID,
                "question": QUESTION,
                "reference_answer": REFERENCE,
            }
        ]
        groundedness = GroundednessEvaluationResult(
            score=5,
            reason="所有事实均有检索支持。",
            claims=(
                GroundingClaim(
                    "列表可以修改",
                    "supported",
                    "列表可以修改",
                    "列表创建后可以修改元素",
                    "语义支持",
                ),
            ),
        )
        case_result = RagCaseResult(
            rag_answer=RagAnswer(
                text=CANDIDATE,
                duration_ms=123,
                project_root=r"E:\Enterprise AI Helpdesk",
                entrypoint="utils.answer:answer_user",
                intent="IT",
                rewritten_question=QUESTION,
                retrieved_context=REFERENCE,
                need_human=False,
                trace_available=True,
            ),
            correctness=result,
            retrieval=EvaluationResult(score=5, reason="检索完整。"),
            groundedness=groundedness,
            error_type="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_results(path, cases, {QUESTION_ID: case_result})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload[0]["rag_answer"], CANDIDATE)
        self.assertEqual(payload[0]["rag_call"]["duration_ms"], 123)
        self.assertEqual(payload[0]["rag_call"]["transport"], "python")
        self.assertEqual(payload[0]["correctness_score"], 4)
        self.assertEqual(payload[0]["groundedness_score"], 5)
        self.assertEqual(payload[0]["error_type"], "none")
        self.assertEqual(payload[0]["retrieval_score"], 5)
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
            EvaluationResult(score=5, reason="检索完整"),
            GroundednessEvaluationResult(score=5, reason="有依据"),
            "generation",
        )
        failures = check_rag_acceptance(cases, {QUESTION_ID: case_result}, 4)
        self.assertEqual(len(failures), 1)
        self.assertIn("error_type=generation", failures[0])

    def test_error_type_classification_matrix(self) -> None:
        good = EvaluationResult(score=5, reason="通过")
        bad = EvaluationResult(score=0, reason="失败")
        grounded = GroundednessEvaluationResult(score=5, reason="有依据")
        ungrounded = GroundednessEvaluationResult(score=0, reason="无依据")
        self.assertEqual(classify_error_type(good, good, grounded, 4), "none")
        self.assertEqual(classify_error_type(bad, bad, grounded, 4), "retrieval")
        self.assertEqual(classify_error_type(bad, good, grounded, 4), "generation")
        self.assertEqual(classify_error_type(bad, bad, ungrounded, 4), "both")

    def test_http_transport_requires_url(self) -> None:
        args = argparse.Namespace(
            rag_transport="http",
            rag_url="",
            rag_api_key_env="EVALFORGE_RAG_API_KEY",
            rag_timeout=60,
        )
        with self.assertRaisesRegex(RagIntegrationError, "--rag-url"):
            build_rag_adapter(args)


class RagAdapterTests(unittest.TestCase):
    def test_calls_configured_python_entrypoint_from_project_directory(self) -> None:
        module_name = "evalforge_test_rag_api"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{module_name}.py").write_text(
                "from pathlib import Path\n"
                "def route_query(question):\n"
                "    return {'intent': 'IT', 'query_rewrite': question, 'need_human': False}\n"
                "def retrieve(query, intent):\n"
                "    return '检索内容'\n"
                "def generate_answer(query, docs):\n"
                "    return f'{Path.cwd().name}:{query}'\n"
                "def answer(question):\n"
                "    route = route_query(question)\n"
                "    docs = retrieve(route['query_rewrite'], route['intent'])\n"
                "    return generate_answer(route['query_rewrite'], docs)\n",
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
        self.assertEqual(result.intent, "IT")
        self.assertEqual(result.retrieved_context, "检索内容")
        self.assertTrue(result.trace_available)
        self.assertFalse(cache_created)

    def test_rejects_empty_rag_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = PythonRagAdapter(Path(directory), "demo:answer", lambda _: "")
            with self.assertRaisesRegex(RagIntegrationError, "空答案"):
                adapter.answer("测试问题")


class HttpRagAdapterTests(unittest.TestCase):
    def test_posts_question_and_reads_strict_trace_response(self) -> None:
        requests = []

        def fake_post(url, headers, body, timeout):
            requests.append((url, headers, json.loads(body), timeout))
            return {
                "answer": "请重启 VPN 客户端。",
                "retrieved_context": "VPN问题：无法连接VPN请重启客户端或检查账号",
                "intent": "IT",
                "rewritten_question": "VPN 无法连接怎么办",
                "need_human": False,
            }

        adapter = HttpRagAdapter(
            "https://rag.example.com/evaluate",
            api_key="secret-token",
            timeout_seconds=25,
            http_post=fake_post,
        )
        result = adapter.answer("VPN 无法连接怎么办？", question_id="Q002")

        self.assertEqual(requests[0][2]["question_id"], "Q002")
        self.assertEqual(requests[0][2]["question"], "VPN 无法连接怎么办？")
        self.assertEqual(requests[0][1]["Authorization"], "Bearer secret-token")
        self.assertEqual(requests[0][3], 25)
        self.assertEqual(result.transport, "http")
        self.assertEqual(result.intent, "IT")
        self.assertTrue(result.trace_available)

    def test_rejects_missing_retrieved_context(self) -> None:
        adapter = HttpRagAdapter(
            "https://rag.example.com/evaluate",
            http_post=lambda *args: {"answer": "回答"},
        )
        with self.assertRaisesRegex(RagIntegrationError, "retrieved_context"):
            adapter.answer("问题")

    def test_rejects_unknown_response_fields(self) -> None:
        adapter = HttpRagAdapter(
            "https://rag.example.com/evaluate",
            http_post=lambda *args: {
                "answer": "回答",
                "retrieved_context": "资料",
                "debug": "不应进入正式契约",
            },
        )
        with self.assertRaisesRegex(RagIntegrationError, "未知字段"):
            adapter.answer("问题")

    def test_real_local_http_round_trip(self) -> None:
        received: dict = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                received.update(json.loads(self.rfile.read(length)))
                payload = json.dumps(
                    {
                        "answer": "提交发票和审批单。",
                        "retrieved_context": "报销流程：提交发票+审批单",
                        "intent": "FINANCE",
                        "rewritten_question": "报销材料",
                        "need_human": False,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = HttpRagAdapter(
                f"http://127.0.0.1:{server.server_port}/rag",
                timeout_seconds=5,
            )
            result = adapter.answer("报销需要什么材料？", question_id="Q003")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(received["question_id"], "Q003")
        self.assertEqual(result.text, "提交发票和审批单。")
        self.assertEqual(result.retrieved_context, "报销流程：提交发票+审批单")


if __name__ == "__main__":
    unittest.main()
