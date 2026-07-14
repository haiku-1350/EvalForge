import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evalforge.cli import load_test_cases
from evalforge.evaluator import DeepSeekEvaluator, EvaluationError, validate_result
from evalforge.reviewer import review_content, review_result


MODEL_B_ENV = {
    "EVALFORGE_MODEL_B": "independent-model-b",
    "EVALFORGE_MODEL_B_API_URL": "https://model-b.example/v1/chat/completions",
    "EVALFORGE_MODEL_B_API_KEY": "model-b-key",
}


def api_response(result: object) -> dict:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {"choices": [{"message": {"content": content}}]}


class ResultValidationTests(unittest.TestCase):
    def test_accepts_valid_schema(self) -> None:
        result = validate_result(
            {
                "score": 4,
                "reason": "核心内容正确，但遗漏一个细节。",
                "keywords": ["核心内容"],
            }
        )
        self.assertEqual(result.score, 4)
        self.assertEqual(result.keywords, ("核心内容",))

    def test_rejects_boolean_score(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result(
                {"score": True, "reason": "非法分数", "keywords": ["关键词"]}
            )

    def test_rejects_out_of_range_score(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result(
                {"score": 6, "reason": "非法分数", "keywords": ["关键词"]}
            )

    def test_rejects_empty_reason(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result({"score": 3, "reason": "  ", "keywords": ["关键词"]})

    def test_rejects_extra_fields(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result(
                {
                    "score": 3,
                    "reason": "理由",
                    "keywords": ["关键词"],
                    "extra": 1,
                }
            )


class PythonReviewerTests(unittest.TestCase):
    def test_accepts_matching_score_and_keyword_coverage(self) -> None:
        report = review_result(
            {
                "score": 4,
                "reason": "覆盖主要内容。",
                "keywords": ["404", "资源找不到", "500", "内部错误"],
            },
            "404 表示资源找不到，500 表示服务器内部错误。",
            "404 是资源找不到；500 是内部错误。",
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.coverage_ratio, 1.0)

    def test_rejects_keyword_not_extracted_from_reference(self) -> None:
        report = review_result(
            {"score": 1, "reason": "错误。", "keywords": ["不存在的词"]},
            "参考答案",
            "待评答案",
        )
        self.assertFalse(report.passed)
        self.assertIn("不在参考答案", report.issues[-1])

    def test_flags_high_score_with_low_coverage(self) -> None:
        report = review_result(
            {
                "score": 5,
                "reason": "声称完整。",
                "keywords": ["原子性", "一致性", "隔离性", "持久性"],
            },
            "原子性、一致性、隔离性、持久性",
            "只说明了原子性和持久性",
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.coverage_ratio, 0.5)
        self.assertTrue(any(issue.startswith("评分冲突：") for issue in report.issues))

    def test_low_score_is_allowed_even_when_keywords_are_repeated(self) -> None:
        report = review_result(
            {"score": 0, "reason": "结论与参考答案相反。", "keywords": ["可变"]},
            "列表可变",
            "列表不是可变的",
        )
        self.assertTrue(report.passed)

    def test_rejects_non_json_content(self) -> None:
        report = review_content("not json", "参考答案", "待评答案")
        self.assertFalse(report.passed)
        self.assertIsNone(report.score)
        self.assertIn("不是合法 JSON", report.issues[0])

    def test_rejects_punctuation_only_keyword(self) -> None:
        report = review_result(
            {"score": 1, "reason": "无关。", "keywords": ["..."]},
            "参考答案",
            "待评答案",
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("只包含空白或标点" in issue for issue in report.issues))


class EvaluatorTests(unittest.TestCase):
    def test_requires_environment_variable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EvaluationError, "DEEPSEEK_API_KEY"):
                DeepSeekEvaluator()

    def test_parses_and_reviews_nested_api_response(self) -> None:
        def fake_post(url, headers, body, timeout):
            request_body = json.loads(body)
            self.assertEqual(request_body["model"], "deepseek-v4-flash")
            self.assertEqual(request_body["temperature"], 0)
            self.assertEqual(request_body["response_format"], {"type": "json_object"})
            self.assertNotIn("test-key", body.decode("utf-8"))
            return api_response(
                {
                    "score": 5,
                    "reason": "语义一致且内容完整。",
                    "keywords": ["参考答案"],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            evaluator = DeepSeekEvaluator(http_post=fake_post)
            result = evaluator.evaluate("问题", "参考答案", "参考答案")
        self.assertEqual(result.score, 5)
        self.assertFalse(result.needs_review)
        self.assertEqual(result.attempts, 1)

    def test_retries_with_python_feedback_then_accepts_correction(self) -> None:
        responses = [
            {
                "score": 5,
                "reason": "错误地判为完整。",
                "keywords": ["原子性", "一致性"],
            },
            {
                "score": 3,
                "reason": "只覆盖了一半关键词。",
                "keywords": ["原子性", "一致性"],
            },
        ]
        request_bodies = []

        def fake_post(url, headers, body, timeout):
            request_bodies.append(json.loads(body))
            return api_response(responses.pop(0))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                "ACID 是什么？", "原子性和一致性", "只提到原子性"
            )

        self.assertEqual(result.score, 3)
        self.assertEqual(result.attempts, 2)
        self.assertFalse(result.needs_review)
        retry_prompt = request_bodies[1]["messages"][1]["content"]
        self.assertIn("Python 复核", retry_prompt)
        self.assertIn("评分冲突", retry_prompt)

    def test_uses_model_b_result_when_blind_review_passes(self) -> None:
        responses = [
            {
                "score": 5,
                "reason": "模型 A 首次虚高。",
                "keywords": ["原子性", "一致性"],
            },
            {
                "score": 5,
                "reason": "模型 A 纠正后仍然虚高。",
                "keywords": ["原子性", "一致性"],
            },
            {
                "score": 3,
                "reason": "模型 B 判断只覆盖一半。",
                "keywords": ["原子性", "一致性"],
            },
        ]
        requests = []

        def fake_post(url, headers, body, timeout):
            requests.append((url, headers, json.loads(body)))
            return api_response(responses.pop(0))

        environment = {"DEEPSEEK_API_KEY": "test-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                "ACID 是什么？", "原子性和一致性", "只提到原子性"
            )

        self.assertEqual(result.score, 3)
        self.assertFalse(result.needs_review)
        self.assertEqual(result.stage, "model_b_blind")
        self.assertEqual(result.model, "independent-model-b")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(
            [attempt.stage for attempt in result.review_history],
            ["model_a_initial", "model_a_correction", "model_b_blind"],
        )
        self.assertEqual(
            requests[2][0], "https://model-b.example/v1/chat/completions"
        )
        self.assertEqual(requests[2][1]["Authorization"], "Bearer model-b-key")
        model_b_body = requests[2][2]
        self.assertEqual(model_b_body["model"], "independent-model-b")
        self.assertNotIn("thinking", model_b_body)
        blind_prompt = model_b_body["messages"][1]["content"]
        self.assertNotIn("Python 复核", blind_prompt)
        self.assertNotIn("上一次输出", blind_prompt)
        self.assertNotIn("模型 A", blind_prompt)

    def test_marks_needs_review_only_after_model_b_also_fails(self) -> None:
        requests = []

        def fake_post(url, headers, body, timeout):
            requests.append((url, json.loads(body)))
            return api_response(
                {
                    "score": 5,
                    "reason": "持续给出虚高分。",
                    "keywords": ["原子性", "一致性"],
                }
            )

        environment = {"DEEPSEEK_API_KEY": "test-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                "ACID 是什么？", "原子性和一致性", "只提到原子性"
            )

        self.assertEqual(len(requests), 3)
        self.assertEqual(
            [request[1]["model"] for request in requests],
            ["deepseek-v4-flash", "deepseek-v4-flash", "independent-model-b"],
        )
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.needs_review)
        self.assertEqual(result.stage, "model_b_blind")
        self.assertTrue(any("评分冲突" in issue for issue in result.review_issues))

    def test_malformed_output_is_handed_to_human_after_model_b_fails(self) -> None:
        def fake_post(url, headers, body, timeout):
            return api_response("not json")

        environment = {"DEEPSEEK_API_KEY": "test-key", **MODEL_B_ENV}
        with patch.dict(os.environ, environment, clear=True):
            result = DeepSeekEvaluator(http_post=fake_post).evaluate(
                "问题", "参考答案", "待评答案"
            )

        self.assertIsNone(result.score)
        self.assertTrue(result.needs_review)
        self.assertEqual(result.attempts, 3)

    def test_model_b_configuration_is_required_only_when_blind_review_runs(self) -> None:
        def fake_post(url, headers, body, timeout):
            return api_response(
                {
                    "score": 5,
                    "reason": "持续给出虚高分。",
                    "keywords": ["原子性", "一致性"],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            evaluator = DeepSeekEvaluator(http_post=fake_post)
            with self.assertRaisesRegex(EvaluationError, "模型 B 尚未配置"):
                evaluator.evaluate(
                    "ACID 是什么？", "原子性和一致性", "只提到原子性"
                )


class TestDataTests(unittest.TestCase):
    def test_bundled_data_has_required_shape(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "test_cases.json"
        cases = load_test_cases(path)
        self.assertGreaterEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(
                {answer["answer_type"] for answer in case["answers"]},
                {"correct", "partial", "incorrect"},
            )


if __name__ == "__main__":
    unittest.main()
