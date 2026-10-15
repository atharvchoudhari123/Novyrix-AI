from .context import build_system_prompt, build_tool_system_prompt, attach_files, detect_mode
from .agent import registry
from .orchestrator import AgentOrchestrator
from .memory import memory
from .model_registry import get_model, get_checkpoint, normalize_model_id
from .runtime import runtime


class NovyrixEngine:
    def __init__(self):
        self.orchestrator = AgentOrchestrator()

    def complete(
        self,
        model_id,
        messages,
        files=None,
        mode="chat",
        max_new_tokens=512,
        plugins=None,
    ):
        # Accept legacy Novyrix IDs while using canonical Novyrix IDs internally.
        canonical_model_id = normalize_model_id(model_id)
        model = get_model(canonical_model_id)

        if model is None:
            raise ValueError(f"Unknown Novyrix model: {model_id}")

        checkpoint = get_checkpoint(canonical_model_id)
        normalized = memory.normalize(messages)

        user_message = ""
        for message in reversed(normalized):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if mode in (None, "", "chat", "auto"):
            mode = detect_mode(user_message, "auto")

        tool_results = self.orchestrator.maybe_tools(canonical_model_id, user_message)
        if tool_results:
            user_message = self.orchestrator.enrich_prompt(user_message, tool_results)
            normalized[-1]["content"] = user_message

        system = build_tool_system_prompt(
            model["display_name"],
            mode=mode,
            tools=registry.descriptions(),
            user_message=user_message,
        )

        normalized = attach_files(normalized, files or [])

        final_messages = [
            {"role": "system", "content": system},
            *normalized,
        ]

        return runtime.generate(
            checkpoint=checkpoint,
            messages=final_messages,
            max_new_tokens=max_new_tokens,
        )

    def stream(
        self,
        model_id,
        messages,
        files=None,
        mode="chat",
        max_new_tokens=512,
        plugins=None,
    ):
        canonical_model_id = normalize_model_id(model_id)
        model = get_model(canonical_model_id)
        if model is None:
            raise ValueError(f"Unknown Novyrix model: {model_id}")

        checkpoint = get_checkpoint(canonical_model_id)
        normalized = memory.normalize(messages)
        user_message = ""
        for message in reversed(normalized):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if mode in (None, "", "chat", "auto"):
            mode = detect_mode(user_message, "auto")

        tool_results = self.orchestrator.maybe_tools(canonical_model_id, user_message)
        if tool_results:
            user_message = self.orchestrator.enrich_prompt(user_message, tool_results)
            normalized[-1]["content"] = user_message

        system = build_tool_system_prompt(
            model["display_name"],
            mode=mode,
            tools=registry.descriptions(),
            user_message=user_message,
        )
        normalized = attach_files(normalized, files or [])
        final_messages = [{"role": "system", "content": system}, *normalized]

        yield from runtime.stream(
            checkpoint=checkpoint,
            messages=final_messages,
            max_new_tokens=max_new_tokens,
        )



engine = NovyrixEngine()
