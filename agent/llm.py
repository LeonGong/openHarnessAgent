from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class LLMConfig:
    backend: str = "rule_based"
    model: str = "rule-based-v1"
    temperature: float = 0.0
    max_tokens: int = 1024


class LLMConfigLoader:
    """Load LLM configuration from JSON file and environment variables.

    Priority (high -> low):
    1. Environment variables with `HARNESS_LLM_` prefix.
    2. JSON file pointed by `HARNESS_LLM_CONFIG`.
    3. Defaults from `LLMConfig`.
    """

    ENV_PREFIX = "HARNESS_LLM_"

    @classmethod
    def load(cls) -> LLMConfig:
        config = LLMConfig()

        config_path = os.getenv("HARNESS_LLM_CONFIG")
        if config_path:
            cls._merge_from_file(config, Path(config_path))

        cls._merge_from_env(config)
        return config

    @classmethod
    def _merge_from_file(cls, config: LLMConfig, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("LLM config file must be a JSON object")

        cls._apply_dict(config, payload)

    @classmethod
    def _merge_from_env(cls, config: LLMConfig) -> None:
        env_data: dict[str, Any] = {}

        if backend := os.getenv(f"{cls.ENV_PREFIX}BACKEND"):
            env_data["backend"] = backend
        if model := os.getenv(f"{cls.ENV_PREFIX}MODEL"):
            env_data["model"] = model
        if temperature := os.getenv(f"{cls.ENV_PREFIX}TEMPERATURE"):
            env_data["temperature"] = float(temperature)
        if max_tokens := os.getenv(f"{cls.ENV_PREFIX}MAX_TOKENS"):
            env_data["max_tokens"] = int(max_tokens)

        cls._apply_dict(config, env_data)

    @staticmethod
    def _apply_dict(config: LLMConfig, payload: dict[str, Any]) -> None:
        for key in ("backend", "model", "temperature", "max_tokens"):
            if key in payload:
                setattr(config, key, payload[key])


def create_llm(config: LLMConfig) -> LLM:
    backend = config.backend.lower().strip()

    if backend == "rule_based":
        return RuleBasedLLM()

    raise ValueError(f"Unsupported LLM backend: {config.backend}")
