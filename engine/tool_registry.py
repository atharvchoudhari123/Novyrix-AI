from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, function: Callable[..., Any]) -> None:
        self._tools[name] = Tool(name, description, function)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def descriptions(self):
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

registry = ToolRegistry()
