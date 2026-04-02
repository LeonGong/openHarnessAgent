from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return either {'type': 'tool_use', ...} or {'type': 'final', ...}."""


class RuleBasedLLM(LLM):
    """Deterministic mock LLM for demo/testing."""

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        _ = tools
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)

        if not last_user:
            return {"type": "final", "content": "No input."}

        task = str(last_user["content"]).lower()

        if not last_tool and any(k in task for k in ("read", "读取", "summarize", "总结")):
            path = self._extract_path(str(last_user["content"]))
            if path:
                return {"type": "tool_use", "name": "file_read", "args": {"path": path}}
            return {"type": "final", "content": "请在任务中用引号提供文件路径。"}

        if last_tool and last_tool.get("name") == "file_read":
            text = last_tool.get("content", "")
            snippet = repr(str(text))
            code = (
                f"text = {snippet}\n"
                "lines = [x for x in text.splitlines() if x.strip()]\n"
                "print('line_count=' + str(len(lines)))\n"
                "print('char_count=' + str(len(text)))"
            )
            return {"type": "tool_use", "name": "python_exec", "args": {"code": code}}

        if last_tool and last_tool.get("name") == "python_exec":
            file_output = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "tool" and m.get("name") == "file_read"),
                "",
            )
            preview = "\n".join(str(file_output).splitlines()[:3])
            return {
                "type": "final",
                "content": f"任务完成。\n统计:\n{last_tool.get('content', '')}\n摘要预览:\n{preview}",
            }

        return {"type": "final", "content": "任务完成。"}

    @staticmethod
    def _extract_path(text: str) -> str | None:
        match = re.search(r"[\"'`]([^\"'`]+)[\"'`]", text) or re.search(r"“([^”]+)”", text)
        if match:
            return match.group(1).strip()
        return None
