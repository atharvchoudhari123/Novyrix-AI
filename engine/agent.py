from __future__ import annotations

import re
from dataclasses import dataclass

from .tool_executor import executor
from .tool_registry import registry
from .tools import (
    analyze_traceback, calculate, check_syntax, edit_file, fetch_webpage,
    list_files, read_file, run_tests, search_files, search_web, write_file,
)

def _register_defaults():
    registry.register("calculator", "Evaluate arithmetic safely.", calculate)
    registry.register("list_files", "List files in the LumaCore workspace.", list_files)
    registry.register("read_file", "Read a UTF-8 text file in the workspace.", read_file)
    registry.register("write_file", "Create or replace a text file in the workspace.", write_file)
    registry.register("edit_file", "Replace one exact text span in a workspace file.", edit_file)
    registry.register("search_files", "Search workspace text files.", search_files)
    registry.register("analyze_traceback", "Summarize a Python traceback.", analyze_traceback)
    registry.register("check_syntax", "Check Python syntax.", check_syntax)
    registry.register("run_tests", "Run pytest in the workspace.", run_tests)
    registry.register("web_search", "Search the web for current context.", search_web)
    registry.register("fetch_webpage", "Fetch a webpage over HTTP(S).", fetch_webpage)

_register_defaults()

@dataclass
class ToolDecision:
    tool: str
    arguments: dict
    reason: str

class LumaCoreAgent:
    def choose_tool(self, text: str):
        lower = (text or "").lower()
        match = re.match(r"^(?:calculate|compute)s+(.+)$", lower)
        if match:
            return ToolDecision("calculator", {"expression": match.group(1)}, "Explicit calculation request.")
        if "list files" in lower or "show files" in lower:
            return ToolDecision("list_files", {}, "Workspace file listing requested.")
        return None

    def run(self, text: str):
        decision = self.choose_tool(text)
        if decision is None:
            return None
        return {
            "tool": decision.tool,
            "reason": decision.reason,
            "result": executor.execute(decision.tool, **decision.arguments),
        }

agent = LumaCoreAgent()
