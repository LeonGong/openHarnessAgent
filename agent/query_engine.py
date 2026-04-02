from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.context_manager import ContextManager
from agent.llm import LLM
from agent.permission import PermissionSystem
from agent.tool_registry import ToolRegistry


@dataclass
class ExecutionResult:
    final_answer: str
    turns_used: int
    trace: list[str] = field(default_factory=list)


class QueryEngine:
    def __init__(
        self,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        llm: LLM,
        permission_system: PermissionSystem,
        max_turns: int = 8,
    ) -> None:
        self.context_manager = context_manager
        self.tool_registry = tool_registry
        self.llm = llm
        self.permission_system = permission_system
        self.max_turns = max_turns

    def run(self, user_input: str) -> ExecutionResult:
        trace: list[str] = []
        self.context_manager.add_user(user_input)

        for turn in range(1, self.max_turns + 1):
            model_output = self.llm.generate(
                messages=self.context_manager.build(),
                tools=[spec.__dict__ for spec in self.tool_registry.specs()],
            )
            output_type = model_output.get("type")
            trace.append(f"turn={turn}, model_output={model_output}")

            if output_type == "final":
                final_text = str(model_output.get("content", ""))
                self.context_manager.add_assistant(final_text)
                return ExecutionResult(final_answer=final_text, turns_used=turn, trace=trace)

            if output_type != "tool_use":
                fallback = f"Invalid model output: {model_output}"
                self.context_manager.add_assistant(fallback)
                return ExecutionResult(final_answer=fallback, turns_used=turn, trace=trace)

            tool_name = str(model_output.get("name", ""))
            tool_args = dict(model_output.get("args", {}))
            self.context_manager.add_tool_use(tool_name, tool_args)

            if not self.permission_system.can_use_tool(tool_name, tool_args):
                denied = f"Permission denied: {tool_name}"
                self.context_manager.add_tool_result(tool_name, denied)
                trace.append(denied)
                continue

            tool = self.tool_registry.get(tool_name)
            if tool is None:
                unknown = f"Unknown tool: {tool_name}"
                self.context_manager.add_tool_result(tool_name, unknown)
                trace.append(unknown)
                continue

            tool_result = tool.run({"turn": turn}, tool_args)
            self.context_manager.add_tool_result(tool_name, tool_result)
            trace.append(f"tool={tool_name}, result={tool_result[:120]}")

        timeout_msg = f"Stopped: reached max_turns={self.max_turns}."
        self.context_manager.add_assistant(timeout_msg)
        return ExecutionResult(final_answer=timeout_msg, turns_used=self.max_turns, trace=trace)
