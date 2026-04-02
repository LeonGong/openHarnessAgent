from __future__ import annotations


class PermissionSystem:
    """Simplified permission gate. Default allow-all, easy to override."""

    def can_use_tool(self, tool_name: str, args: dict) -> bool:
        _ = (tool_name, args)
        return True
