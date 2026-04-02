from __future__ import annotations

from pathlib import Path

from .base import Tool


class FileReadTool(Tool):
    name = "file_read"
    description = "Read UTF-8 text from a local path."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }

    def run(self, context: dict, args: dict) -> str:
        _ = context
        target = Path(args["path"]).expanduser()
        max_chars = int(args.get("max_chars", 4000))

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: failed to read '{target}': {exc}"

        if len(content) > max_chars:
            return content[:max_chars] + "\n...[truncated]"
        return content
