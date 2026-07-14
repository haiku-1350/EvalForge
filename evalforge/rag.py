from __future__ import annotations

import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator


DEFAULT_RAG_PROJECT = Path(r"E:\Enterprise AI Helpdesk")
DEFAULT_RAG_ENTRYPOINT = "utils.answer:answer_user"
_WORKING_DIRECTORY_LOCK = RLock()
HTTP_RESPONSE_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
HTTP_REQUIRED_RESPONSE_FIELDS = {"answer", "retrieved_context"}
HTTP_OPTIONAL_RESPONSE_FIELDS = {
    "intent",
    "rewritten_question",
    "need_human",
}


class RagIntegrationError(RuntimeError):
    """Raised when the configured RAG Python interface cannot be called."""


@dataclass(frozen=True)
class RagAnswer:
    text: str
    duration_ms: int
    project_root: str | None
    entrypoint: str
    intent: str | None = None
    rewritten_question: str | None = None
    retrieved_context: str = ""
    need_human: bool | None = None
    trace_available: bool = False
    transport: str = "python"


RagHttpPost = Callable[[str, dict[str, str], bytes, int], dict[str, Any]]


def _default_rag_http_post(
    url: str, headers: dict[str, str], body: bytes, timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(HTTP_RESPONSE_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RagIntegrationError(
            f"外部 RAG HTTP 接口返回 {exc.code}：{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RagIntegrationError(f"无法连接外部 RAG HTTP 接口：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RagIntegrationError("调用外部 RAG HTTP 接口超时") from exc
    if len(payload) > HTTP_RESPONSE_MAX_BYTES:
        raise RagIntegrationError("外部 RAG HTTP 响应超过 2 MiB 限制")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RagIntegrationError("外部 RAG HTTP 响应不是合法 UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise RagIntegrationError("外部 RAG HTTP 响应必须是 JSON 对象")
    return decoded


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    with _WORKING_DIRECTORY_LOCK:
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)


class PythonRagAdapter:
    """Call a local RAG system through a ``module:function`` Python interface."""

    def __init__(
        self,
        project_root: Path | str = DEFAULT_RAG_PROJECT,
        entrypoint: str = DEFAULT_RAG_ENTRYPOINT,
        answer_callable: Callable[[str], object] | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.entrypoint = entrypoint
        if not self.project_root.is_dir():
            raise RagIntegrationError(f"RAG 项目目录不存在：{self.project_root}")
        self._answer_callable = answer_callable or self._load_callable()

    def _load_callable(self) -> Callable[[str], object]:
        module_name, separator, function_name = self.entrypoint.partition(":")
        if not separator or not module_name or not function_name:
            raise RagIntegrationError(
                "RAG 入口格式必须是 module:function，例如 utils.answer:answer_user"
            )

        project_text = str(self.project_root)
        sys.path.insert(0, project_text)
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            with _working_directory(self.project_root):
                module = importlib.import_module(module_name)
        except Exception as exc:
            raise RagIntegrationError(
                f"无法导入 RAG Python 入口 {self.entrypoint}：{exc}"
            ) from exc
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            try:
                sys.path.remove(project_text)
            except ValueError:
                pass

        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RagIntegrationError(f"RAG 模块没有可验证的文件路径：{module_name}")
        try:
            Path(module_file).resolve().relative_to(self.project_root)
        except ValueError as exc:
            raise RagIntegrationError(
                f"导入的模块不属于指定 RAG 项目：{module_file}"
            ) from exc

        function = getattr(module, function_name, None)
        if not callable(function):
            raise RagIntegrationError(f"RAG 入口不是可调用函数：{self.entrypoint}")
        return function

    def answer(self, question: str, question_id: str | None = None) -> RagAnswer:
        if not isinstance(question, str) or not question.strip():
            raise RagIntegrationError("发送给 RAG 的问题不能为空")
        started = time.perf_counter()
        trace: dict[str, object] = {
            "intent": None,
            "rewritten_question": None,
            "retrieved_context": "",
            "need_human": None,
            "trace_available": False,
        }
        try:
            with _working_directory(self.project_root):
                raw_answer = self._call_with_trace(question, trace)
        except Exception as exc:
            raise RagIntegrationError(f"RAG 回答调用失败：{exc}") from exc
        duration_ms = round((time.perf_counter() - started) * 1000)
        if isinstance(raw_answer, dict):
            answer_text = raw_answer.get("answer")
            trace.update(
                {
                    "intent": raw_answer.get("intent"),
                    "rewritten_question": raw_answer.get("rewritten_question"),
                    "retrieved_context": raw_answer.get("retrieved_context", ""),
                    "need_human": raw_answer.get("need_human"),
                    "trace_available": True,
                }
            )
        else:
            answer_text = raw_answer
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise RagIntegrationError("RAG Python 接口返回了空答案或非字符串")
        retrieved_context = trace["retrieved_context"]
        if not isinstance(retrieved_context, str):
            raise RagIntegrationError("RAG 检索内容必须是字符串")
        return RagAnswer(
            text=answer_text.strip(),
            duration_ms=duration_ms,
            project_root=str(self.project_root),
            entrypoint=self.entrypoint,
            intent=trace["intent"] if isinstance(trace["intent"], str) else None,
            rewritten_question=trace["rewritten_question"]
            if isinstance(trace["rewritten_question"], str)
            else None,
            retrieved_context=retrieved_context.strip(),
            need_human=trace["need_human"]
            if isinstance(trace["need_human"], bool)
            else None,
            trace_available=bool(trace["trace_available"]),
            transport="python",
        )

    def _call_with_trace(
        self, question: str, trace: dict[str, object]
    ) -> object:
        function_globals = getattr(self._answer_callable, "__globals__", None)
        if not isinstance(function_globals, dict):
            return self._answer_callable(question)
        route_function = function_globals.get("route_query")
        retrieve_function = function_globals.get("retrieve")
        if not callable(route_function) or not callable(retrieve_function):
            return self._answer_callable(question)

        def traced_route(text: str) -> object:
            route = route_function(text)
            if isinstance(route, dict):
                trace["intent"] = route.get("intent")
                trace["rewritten_question"] = route.get("query_rewrite")
                trace["need_human"] = route.get("need_human")
            return route

        def traced_retrieve(query: str, intent: str) -> object:
            context = retrieve_function(query, intent)
            trace["retrieved_context"] = context
            return context

        trace["trace_available"] = True
        function_globals["route_query"] = traced_route
        function_globals["retrieve"] = traced_retrieve
        try:
            return self._answer_callable(question)
        finally:
            function_globals["route_query"] = route_function
            function_globals["retrieve"] = retrieve_function


class HttpRagAdapter:
    """Call an external RAG system through a strict JSON-over-HTTP contract."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str = "",
        timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
        http_post: RagHttpPost | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RagIntegrationError("RAG HTTP URL 必须是有效的 http:// 或 https:// 地址")
        if not 1 <= timeout_seconds <= 300:
            raise RagIntegrationError("RAG HTTP 超时必须在 1 到 300 秒之间")
        self.url = url
        self.entrypoint = url
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._http_post = http_post or _default_rag_http_post

    def answer(self, question: str, question_id: str | None = None) -> RagAnswer:
        if not isinstance(question, str) or not question.strip():
            raise RagIntegrationError("发送给 RAG 的问题不能为空")
        request_payload = {
            "question_id": question_id or "Q",
            "question": question.strip(),
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        response = self._http_post(
            self.url,
            headers,
            body,
            self.timeout_seconds,
        )
        duration_ms = round((time.perf_counter() - started) * 1000)
        return self._review_response(response, duration_ms)

    def _review_response(
        self, response: dict[str, Any], duration_ms: int
    ) -> RagAnswer:
        if not isinstance(response, dict):
            raise RagIntegrationError("RAG HTTP 响应必须是 JSON 对象")
        fields = set(response)
        missing = sorted(HTTP_REQUIRED_RESPONSE_FIELDS - fields)
        extra = sorted(
            fields - HTTP_REQUIRED_RESPONSE_FIELDS - HTTP_OPTIONAL_RESPONSE_FIELDS
        )
        if missing:
            raise RagIntegrationError("RAG HTTP 响应缺少字段：" + ", ".join(missing))
        if extra:
            raise RagIntegrationError("RAG HTTP 响应包含未知字段：" + ", ".join(extra))

        answer = response.get("answer")
        retrieved_context = response.get("retrieved_context")
        intent = response.get("intent")
        rewritten_question = response.get("rewritten_question")
        need_human = response.get("need_human")
        if not isinstance(answer, str) or not answer.strip():
            raise RagIntegrationError("RAG HTTP 响应的 answer 必须是非空字符串")
        if not isinstance(retrieved_context, str):
            raise RagIntegrationError("RAG HTTP 响应的 retrieved_context 必须是字符串")
        if intent is not None and not isinstance(intent, str):
            raise RagIntegrationError("RAG HTTP 响应的 intent 必须是字符串或 null")
        if rewritten_question is not None and not isinstance(rewritten_question, str):
            raise RagIntegrationError(
                "RAG HTTP 响应的 rewritten_question 必须是字符串或 null"
            )
        if need_human is not None and not isinstance(need_human, bool):
            raise RagIntegrationError("RAG HTTP 响应的 need_human 必须是布尔值或 null")

        return RagAnswer(
            text=answer.strip(),
            duration_ms=duration_ms,
            project_root=None,
            entrypoint=self.url,
            intent=intent.strip() if isinstance(intent, str) else None,
            rewritten_question=rewritten_question.strip()
            if isinstance(rewritten_question, str)
            else None,
            retrieved_context=retrieved_context.strip(),
            need_human=need_human,
            trace_available=True,
            transport="http",
        )
