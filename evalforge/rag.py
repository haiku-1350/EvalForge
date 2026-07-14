from __future__ import annotations

import importlib
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator


DEFAULT_RAG_PROJECT = Path(r"E:\Enterprise AI Helpdesk")
DEFAULT_RAG_ENTRYPOINT = "utils.answer:answer_user"
_WORKING_DIRECTORY_LOCK = RLock()


class RagIntegrationError(RuntimeError):
    """Raised when the configured RAG Python interface cannot be called."""


@dataclass(frozen=True)
class RagAnswer:
    text: str
    duration_ms: int
    project_root: str
    entrypoint: str


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

    def answer(self, question: str) -> RagAnswer:
        if not isinstance(question, str) or not question.strip():
            raise RagIntegrationError("发送给 RAG 的问题不能为空")
        started = time.perf_counter()
        try:
            with _working_directory(self.project_root):
                raw_answer = self._answer_callable(question)
        except Exception as exc:
            raise RagIntegrationError(f"RAG 回答调用失败：{exc}") from exc
        duration_ms = round((time.perf_counter() - started) * 1000)
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            raise RagIntegrationError("RAG Python 接口返回了空答案或非字符串")
        return RagAnswer(
            text=raw_answer.strip(),
            duration_ms=duration_ms,
            project_root=str(self.project_root),
            entrypoint=self.entrypoint,
        )
