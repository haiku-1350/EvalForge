import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evalforge.cli import load_test_cases
from evalforge.evaluator import DeepSeekEvaluator, EvaluationError, validate_result


class ResultValidationTests(unittest.TestCase):
    def test_accepts_valid_result(self) -> None:
        result = validate_result({"score": 4, "reason": "核心内容正确，但遗漏一个细节。"})
        self.assertEqual(result.score, 4)

    def test_rejects_boolean_score(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result({"score": True, "reason": "非法分数"})

    def test_rejects_out_of_range_score(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result({"score": 6, "reason": "非法分数"})

    def test_rejects_empty_reason(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result({"score": 3, "reason": "  "})

    def test_rejects_extra_fields(self) -> None:
        with self.assertRaises(EvaluationError):
            validate_result({"score": 3, "reason": "理由", "extra": 1})


class EvaluatorTests(unittest.TestCase):
    def test_requires_environment_variable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EvaluationError, "DEEPSEEK_API_KEY"):
                DeepSeekEvaluator()

    def test_parses_nested_api_response(self) -> None:
        def fake_post(url, headers, body, timeout):
            request_body = json.loads(body)
            self.assertEqual(request_body["model"], "deepseek-v4-flash")
            self.assertEqual(request_body["temperature"], 0)
            self.assertEqual(request_body["response_format"], {"type": "json_object"})
            self.assertNotIn("test-key", body.decode("utf-8"))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"score": 5, "reason": "语义一致且内容完整。"},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            evaluator = DeepSeekEvaluator(http_post=fake_post)
            result = evaluator.evaluate("问题", "参考答案", "待评估答案")
        self.assertEqual(result.score, 5)


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
