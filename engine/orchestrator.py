from __future__ import annotations

from .advanced_agent import advanced_planner
from .agent import agent
from .agent_policy import allowed_tools


class AgentOrchestrator:
    """Coordinates planning and safe, deterministic tool execution."""

    def plan(self, user_text: str):
        return advanced_planner.plan(user_text)

    def maybe_tools(self, model_id: str, user_text: str):
        allowed = allowed_tools(model_id)
        decisions = [
            decision
            for decision in agent.choose_tools(user_text)
            if decision.tool in allowed
        ]
        if not decisions:
            return []
        return agent.run(decisions)

    def maybe_tool(self, model_id: str, user_text: str):
        results = self.maybe_tools(model_id, user_text)
        return results[0] if results else None

    def enrich_prompt(self, user_text: str, tool_results):
        if not tool_results:
            return user_text

        if isinstance(tool_results, dict):
            tool_results = [tool_results]

        evidence = "\n\n".join(
            f"[{item['tool']}] {item['result']!r}"
            for item in tool_results
        )

        return (
            user_text
            + "\n\nNovyrix Advanced tool results. Treat these as evidence; "
            "do not claim anything beyond what they establish:\n"
            + evidence
        )


orchestrator = AgentOrchestrator()
