from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Message = dict[str, Any]


@dataclass
class ContextManager:
    system_prompt: str
    memory_blocks: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_use(self, name: str, args: dict[str, Any]) -> None:
        self.messages.append({"role": "assistant", "content": {"type": "tool_use", "name": name, "args": args}})

    def add_tool_result(self, name: str, result: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "name": name,
                "content": result,
            }
        )

    def build(self) -> list[Message]:
        context: list[Message] = [{"role": "system", "content": self.system_prompt}]
        if self.memory_blocks:
            context.append({"role": "system", "content": "Memory:\n" + "\n".join(self.memory_blocks)})
        context.extend(self.messages)
        return context
