from __future__ import annotations

import re
from dataclasses import dataclass

from .advanced_agent import advanced_planner
from .tool_executor import executor
from .tool_registry import registry
from .tools import (
    analyze_traceback, calculate, check_syntax, edit_file, fetch_webpage,
    list_files, read_file, run_tests, search_files, search_web, write_file,
)


def _register_defaults():
    registry.register("calculator", "Evaluate arithmetic safely.", calculate)
    registry.register("list_files", "List files in the Novyrix workspace.", list_files)
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


class NovyrixAgent:
    """Lightweight deterministic tool router used by the Advanced beta."""

    def choose_tools(self, text: str) -> list[ToolDecision]:
        value = (text or "").strip()
        lower = value.lower()
        plan = advanced_planner.plan(value)
        decisions: list[ToolDecision] = []

        # Explicit arithmetic requests.
        match = re.match(r"^(?:calculate|compute)\s+(.+)$", lower)
        if match:
            decisions.append(
                ToolDecision(
                    "calculator",
                    {"expression": match.group(1)},
                    "Explicit calculation request.",
                )
            )

        # Explicit workspace listing.
        if "list files" in lower or "show files" in lower:
            decisions.append(
                ToolDecision(
                    "list_files",
                    {},
                    "Workspace file listing requested.",
                )
            )

        # Search a project when the user explicitly asks to find text.
        search_match = re.search(
            r"(?:search|find)(?: for)?(?: the text)?\\s+(.+?)(?:\\s+in\\s+(?:the\\s+)?(?:project|workspace|files))?$",
            value,
            re.IGNORECASE,
        )
        if search_match and ("search files" in lower or "find in" in lower):
            decisions.append(
                ToolDecision(
                    "search_files",
                    {"query": search_match.group(1).strip()},
                    "The user requested a workspace search.",
                )
            )

        # Web research is useful when the request explicitly needs current information.
        if plan.intent == "research" and (
            any(word in lower for word in ("latest", "today", "current", "recent", "search", "research"))
        ):
            decisions.append(
                ToolDecision(
                    "web_search",
                    {"query": value},
                    "The request needs external/current context.",
                )
            )

        # Tracebacks can be analyzed directly without pretending the model executed them.
        if "traceback" in lower or "stack trace" in lower:
            decisions.append(
                ToolDecision(
                    "analyze_traceback",
                    {"traceback_text": value},
                    "The user supplied or referenced a traceback.",
                )
            )

        return decisions

    def run(self, decisions: list[ToolDecision]):
        results = []
        for decision in decisions:
            results.append(
                {
                    "tool": decision.tool,
                    "reason": decision.reason,
                    "result": executor.execute(decision.tool, **decision.arguments),
                }
            )
        return results


agent = NovyrixAgent()
