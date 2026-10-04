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

    _web_explicit_words = (
        "search the web", "search online", "look this up", "look it up",
        "research this", "research online", "find online", "find on the web",
        "check online", "check the docs", "check documentation",
        "official docs", "documentation", "sources", "latest version",
        "current version", "recent version", "what changed", "release notes",
    )

    _technical_web_words = (
        "api", "sdk", "package", "library", "framework", "dependency",
        "npm", "pypi", "pip", "maven", "gradle", "cargo", "nuget",
        "swift package", "pod", "homebrew", "docker image",
    )

    _technical_action_words = (
        "install", "configure", "setup", "set up", "integrate", "implement",
        "use", "upgrade", "migrate", "build", "fix", "debug", "deploy",
    )

    @classmethod
    def _needs_web_research(cls, text: str, intent: str) -> tuple[bool, str]:
        lower = (text or "").lower()

        if any(phrase in lower for phrase in cls._web_explicit_words):
            return True, "The user explicitly requested web research or current documentation."

        # Technical implementation questions sometimes depend on changing APIs,
        # package versions, or framework behavior. Search only when the prompt
        # gives us a concrete technical signal; do not browse for ordinary coding.
        technical_signal = any(word in lower for word in cls._technical_web_words)
        technical_action = any(word in lower for word in cls._technical_action_words)
        version_signal = bool(re.search(r"\\b(?:v?\\d+(?:\\.\\d+){0,2}|20\\d{2})\\b", lower))

        if intent in {"code", "build", "debug", "research"} and technical_signal and (
            technical_action or version_signal
        ):
            return True, "The task may depend on external technical documentation or version-specific behavior."

        # Named platforms/frameworks with explicit integration questions are
        # another good reason to consult current docs.
        integration_terms = (
            "github api", "stripe", "discord api", "roblox api", "openai api",
            "anthropic api", "google api", "aws", "azure", "firebase",
            "supabase", "vercel", "cloudflare",
        )
        if intent in {"code", "build", "debug"} and any(term in lower for term in integration_terms):
            return True, "The task references an external platform whose API or behavior may have changed."

        return False, ""

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
            r"(?:search|find)(?: for)?(?: the text)?\s+(.+?)(?:\s+in\s+(?:the\s+)?(?:project|workspace|files))?$",
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

        # Browse selectively. Ordinary coding does not trigger web access;
        # explicit research, current/version-specific questions, and external
        # technical APIs/packages do.
        needs_web, web_reason = self._needs_web_research(value, plan.intent)
        if needs_web:
            decisions.append(
                ToolDecision(
                    "web_search",
                    {"query": value},
                    web_reason,
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
