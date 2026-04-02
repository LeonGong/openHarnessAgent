from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
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


class OpenAIChatLLM(LLM):
    """
    Minimal OpenAI-compatible Chat Completions adapter.

    Environment variables:
    - OPENAI_API_KEY: required
    - OPENAI_BASE_URL: optional, default https://api.openai.com/v1
    - OPENAI_MODEL: optional, default gpt-4.1-mini
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> OpenAIChatLLM | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "tools": self._to_openai_tools(tools),
            "tool_choice": "auto",
            "temperature": 0,
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"type": "final", "content": f"LLM HTTPError: {exc.code} {detail}"}
        except Exception as exc:
            return {"type": "final", "content": f"LLM request failed: {exc}"}

        data = json.loads(raw)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            fn = tool_calls[0].get("function", {})
            name = str(fn.get("name", ""))
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"raw_arguments": raw_args}
            return {"type": "tool_use", "name": name, "args": args}

        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
            )
        return {"type": "final", "content": str(content)}

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for spec in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return result

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")

            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)

            if role == "tool":
                name = m.get("name", "tool")
                normalized.append(
                    {
                        "role": "user",
                        "content": f"[Tool result from {name}]\n{content}",
                    }
                )
                continue

            if role not in ("system", "user", "assistant"):
                role = "user"
            normalized.append({"role": role, "content": str(content)})
        return normalized
