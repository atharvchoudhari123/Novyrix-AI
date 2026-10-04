from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AdvancedPlan:
    intent: str
    goal: str
    steps: list[str] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)


class AdvancedPlanner:
    """
    Deterministic front-end planner for the Novyrix Advanced beta.

    This does not pretend to replace the language model. Its job is to turn a
    vague user request into a useful execution contract so the model knows
    whether it should answer, code, debug, research, inspect files, or execute
    a multi-step workflow.
    """

    _current_words = (
        "latest", "today", "current", "right now", "recent", "newest",
        "this week", "this month",
    )

    _code_words = (
        "code", "coding", "program", "script", "function", "class", "api",
        "website", "web app", "app", "application", "game", "plugin",
        "html", "css", "javascript", "typescript", "python", "java", "c++",
        "c#", "rust", "go", "swift", "kotlin", "lua", "sql", "bash", "shell",
        "react", "vue", "next.js", "node", "node.js", "roblox", "unity",
        "unreal", "write me", "build me", "make me", "create me",
    )

    _debug_words = (
        "error", "exception", "traceback", "bug", "broken", "crash", "fails",
        "failed", "not working", "doesn't work", "does not work", "fix this",
        "fix my", "debug", "why isn't", "why is this",
    )

    _file_words = (
        "file", "folder", "directory", "repo", "repository", "project",
        "upload", "attached", "attachment", "read this", "inspect this",
        "edit this", "modify this", "change this", "create a file",
    )

    _research_words = (
        "research", "look up", "search", "find out", "compare", "sources",
        "documentation", "docs", "what does", "how does", "why does",
    )

    _math_words = (
        "calculate", "compute", "equation", "solve", "math", "percentage",
        "percent", "convert", "how much is",
    )

    _creation_words = (
        "create", "make", "build", "design", "generate", "write", "develop",
        "implement", "set up", "setup",
    )

    def classify(self, text: str) -> str:
        value = (text or "").strip().lower()

        if not value:
            return "general"

        if self._contains(value, self._debug_words):
            return "debug"

        if self._contains(value, self._current_words) and (
            self._contains(value, self._research_words)
            or self._contains(value, self._code_words)
        ):
            return "research"

        if self._contains(value, self._math_words) and not self._contains(
            value, self._code_words
        ):
            return "math"

        if self._contains(value, self._file_words):
            if self._contains(value, self._creation_words):
                return "build"
            return "files"

        if self._contains(value, self._code_words):
            return "code"

        if self._contains(value, self._research_words):
            return "research"

        if self._contains(value, self._creation_words):
            return "build"

        return "general"

    @staticmethod
    def _contains(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    def plan(self, text: str) -> AdvancedPlan:
        intent = self.classify(text)
        goal = (text or "").strip()

        plans = {
            "general": (
                ["Understand the request", "Answer directly and clearly"],
                [],
            ),
            "code": (
                [
                    "Understand the requested behavior and constraints",
                    "Choose an appropriate implementation",
                    "Produce complete usable code",
                    "Check the solution for obvious errors and missing pieces",
                    "Explain how to use or verify it",
                ],
                ["read_file", "search_files", "write_file", "edit_file"],
            ),
            "build": (
                [
                    "Turn the request into concrete deliverables",
                    "Choose the project structure and implementation",
                    "Create or modify the required files",
                    "Review the result for consistency",
                    "Provide run and verification instructions",
                ],
                ["list_files", "read_file", "write_file", "edit_file", "run_tests"],
            ),
            "debug": (
                [
                    "Identify the failure from the provided evidence",
                    "Locate the likely source of the problem",
                    "Apply the smallest reliable fix",
                    "Verify the fix when a local verification tool is available",
                    "Explain what changed and why",
                ],
                ["read_file", "search_files", "analyze_traceback", "check_syntax", "run_tests"],
            ),
            "research": (
                [
                    "Clarify the exact question being answered",
                    "Gather current or relevant evidence",
                    "Separate verified facts from assumptions",
                    "Synthesize the findings into a useful answer",
                ],
                ["web_search", "fetch_webpage"],
            ),
            "files": (
                [
                    "Inspect the supplied files or workspace",
                    "Identify the relevant content",
                    "Perform the requested file operation",
                    "Verify the resulting state",
                ],
                ["list_files", "read_file", "search_files", "write_file", "edit_file"],
            ),
            "math": (
                ["Translate the request into a calculation", "Calculate it safely", "Show the useful result"],
                ["calculator"],
            ),
        }

        steps, tools = plans[intent]
        return AdvancedPlan(
            intent=intent,
            goal=goal,
            steps=steps,
            tool_hints=tools,
        )

    def prompt_block(self, text: str) -> str:
        plan = self.plan(text)
        lines = [
            "NOVYRIX ADVANCED EXECUTION PLAN",
            f"Intent: {plan.intent}",
            f"User goal: {plan.goal}",
            "Plan:",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(plan.steps, 1))
        if plan.tool_hints:
            lines.append("Relevant tools: " + ", ".join(plan.tool_hints))
        lines.extend(
            [
                "",
                "Execution rules:",
                "- Follow the user's actual goal, not just keywords in the prompt.",
                "- If the request asks for code, write the requested code instead of merely discussing coding.",
                "- If the request asks for a multi-file project, provide the project structure and complete file contents when practical.",
                "- If files or tools are available, use them as evidence and perform requested operations rather than claiming they happened.",
                "- If a tool is unavailable, continue with the best useful answer and clearly state the limitation.",
                "- Do not invent tool results, tests, files, web research, or completed actions.",
                "- Do not copy another AI's personality. Novyrix should be direct, practical, creative, and independently branded.",
            ]
        )
        return "\n".join(lines)


advanced_planner = AdvancedPlanner()
