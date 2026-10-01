from __future__ import annotations

from .tool_registry import registry

class ToolExecutionError(RuntimeError):
    pass

class ToolExecutor:
    def execute(self, name: str, **kwargs):
        tool = registry.get(name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: {name}")
        try:
            return tool.function(**kwargs)
        except Exception as exc:
            raise ToolExecutionError(f"{name}: {exc}") from exc

executor = ToolExecutor()
