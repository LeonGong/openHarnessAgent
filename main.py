from __future__ import annotations

from pathlib import Path

from agent.context_manager import ContextManager
from agent.llm import LLMConfig, LLMConfigLoader, create_llm
from agent.permission import PermissionSystem
from agent.query_engine import QueryEngine
from agent.tool_registry import ToolRegistry
from agent.tools.file_read import FileReadTool
from agent.tools.python_exec import PythonExecTool


def build_engine(llm_config: LLMConfig | None = None) -> QueryEngine:
    llm_config = llm_config or LLMConfigLoader.load()

    context = ContextManager(
        system_prompt=(
            "You are an execution harness. Use tools when needed, then provide a final answer."
        ),
        memory_blocks=[
            "Harness focuses on execution loop, not chat UX.",
            "Use explicit tool calls with {type, name, args}.",
            f"LLM backend={llm_config.backend}, model={llm_config.model}, temp={llm_config.temperature}",
        ],
    )

    registry = ToolRegistry()
    registry.register(FileReadTool())
    registry.register(PythonExecTool())

    return QueryEngine(
        context_manager=context,
        tool_registry=registry,
        llm=create_llm(llm_config),
        permission_system=PermissionSystem(),
        max_turns=6,
    )


def main() -> None:
    sample = Path("sample_note.txt")
    sample.write_text(
        "这是一份最小化 Harness 示例。\n"
        "核心能力：执行循环、上下文管理、工具调用、权限钩子。\n"
        "目标：读取文件并生成摘要。\n",
        encoding="utf-8",
    )

    llm_config = LLMConfigLoader.load()
    engine = build_engine(llm_config)
    task = "请读取文件 'sample_note.txt' 并总结关键点。"
    result = engine.run(task)

    print("=== LLM Config ===")
    print(llm_config)

    print("=== Task ===")
    print(task)
    print("\n=== Final Answer ===")
    print(result.final_answer)
    print(f"\n(turns_used={result.turns_used})")

    print("\n=== Execution Trace ===")
    for item in result.trace:
        print("-", item)

    print("\n=== Message Log ===")
    for message in engine.context_manager.build():
        role = message.get("role", "unknown")
        name = message.get("name")
        tag = f"{role}:{name}" if name else role
        print(f"[{tag}] {message.get('content')}")


if __name__ == "__main__":
    main()
