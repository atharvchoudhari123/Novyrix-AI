from __future__ import annotations

from .agent import agent
from .agent_policy import allowed_tools

class AgentOrchestrator:
    def maybe_tool(self, model_id: str, user_text: str):
        decision = agent.choose_tool(user_text)
        if decision is None or decision.tool not in allowed_tools(model_id):
            return None
        return agent.run(user_text)

    def enrich_prompt(self, user_text: str, tool_result):
        if not tool_result:
            return user_text
        return (
            user_text
            + "\n\nLumaCore tool result (use as evidence):\n"
            + repr(tool_result["result"])
        )

orchestrator = AgentOrchestrator()
