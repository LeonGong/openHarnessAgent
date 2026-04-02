from __future__ import annotations

import contextlib
import io

from .base import Tool


class PythonExecTool(Tool):
    name = "python_exec"
    description = "Execute tiny Python snippets and return stdout."
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }

    def run(self, context: dict, args: dict) -> str:
        _ = context
        code = str(args["code"])
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
        }

        stdout = io.StringIO()
        locals_dict: dict = {}
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, {"__builtins__": safe_builtins}, locals_dict)
        except Exception as exc:
            return f"ERROR: python_exec failed: {exc}"

        output = stdout.getvalue().strip()
        return output if output else "(no output)"
