import re

from .context import build_system_prompt, build_tool_system_prompt, attach_files, detect_mode
from .example_retrieval import retrieve_reviewed_answer
from .agent import registry
from .orchestrator import AgentOrchestrator
from .memory import memory
from .model_registry import get_model, get_checkpoint, normalize_model_id
from .runtime import runtime
from .response_quality import (
    UNREADABLE_FALLBACK,
    is_coding_capability_question,
    is_readable_response,
    is_simple_greeting,
)
from .tools.calculator import calculate


_ARITHMETIC_PREFIX = re.compile(
    r"^(?:what\s+is|what's|whats|calculate|compute|solve)\s+",
    re.IGNORECASE,
)
_ARITHMETIC_CHARACTERS = re.compile(r"^[0-9().+*/%\-\s]+$")


def _answer_simple_arithmetic(text):
    """Answer plain arithmetic locally, without web search or LM sampling."""
    expression = _ARITHMETIC_PREFIX.sub("", (text or "").strip())
    expression = re.sub(r"[?.!]+\s*$", "", expression).strip()
    expression = re.sub(
        r"(?<=[0-9.)])\s+times\s+(?=[0-9.(])", "*", expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?<=[0-9.)])\s+multiplied\s+by\s+(?=[0-9.(])", "*", expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?<=[0-9.)])\s+divided\s+by\s+(?=[0-9.(])", "/", expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?<=[0-9.)])\s+plus\s+(?=[0-9.(])", "+", expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?<=[0-9.)])\s+minus\s+(?=[0-9.(])", "-", expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(
        r"(?<=[0-9.)])\s*[x×]\s*(?=[0-9.(])", "*", expression,
        flags=re.IGNORECASE,
    )
    expression = expression.replace("÷", "/").replace("−", "-")
    if not _ARITHMETIC_CHARACTERS.fullmatch(expression):
        return None
    if not any(operator in expression for operator in "+-*/%"):
        return None
    try:
        result = calculate(expression)
    except (ArithmeticError, SyntaxError, ValueError, OverflowError):
        return None
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


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

        normalized = memory.normalize(messages)

        user_message = ""
        for message in reversed(normalized):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        arithmetic_answer = _answer_simple_arithmetic(user_message)
        if arithmetic_answer:
            return arithmetic_answer
        if is_simple_greeting(user_message):
            return "Hi! What can I help you with?"
        if is_coding_capability_question(user_message):
            return "Yes—I can help write, explain, and debug code. Tell me what you want to build and which language you prefer."

        if mode in (None, "", "chat", "auto"):
            mode = detect_mode(user_message, "auto")

        reviewed_answer = retrieve_reviewed_answer(
            canonical_model_id, user_message, files=files, mode=mode
        )
        if reviewed_answer:
            return reviewed_answer

        tool_results = self.orchestrator.maybe_tools(canonical_model_id, user_message)
        research_answer = self.orchestrator.research_answer(tool_results)
        if research_answer and mode == "chat":
            return research_answer
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

        checkpoint = get_checkpoint(canonical_model_id)
        answer = runtime.generate(
            checkpoint=checkpoint,
            messages=final_messages,
            max_new_tokens=min(
                max(1, max_new_tokens),
                int(model.get("max_new_tokens", 512)),
            ),
            temperature=float(model.get("temperature", 0.7)),
        )
        if not tool_results and not is_readable_response(answer):
            return UNREADABLE_FALLBACK
        return answer

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

        normalized = memory.normalize(messages)
        user_message = ""
        for message in reversed(normalized):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        arithmetic_answer = _answer_simple_arithmetic(user_message)
        if arithmetic_answer:
            yield arithmetic_answer
            return
        if is_simple_greeting(user_message):
            yield "Hi! What can I help you with?"
            return
        if is_coding_capability_question(user_message):
            yield "Yes—I can help write, explain, and debug code. Tell me what you want to build and which language you prefer."
            return

        if mode in (None, "", "chat", "auto"):
            mode = detect_mode(user_message, "auto")

        reviewed_answer = retrieve_reviewed_answer(
            canonical_model_id, user_message, files=files, mode=mode
        )
        if reviewed_answer:
            for start in range(0, len(reviewed_answer), 24):
                yield reviewed_answer[start:start + 24]
            return

        tool_results = self.orchestrator.maybe_tools(canonical_model_id, user_message)
        research_answer = self.orchestrator.research_answer(tool_results)
        if research_answer and mode == "chat":
            for start in range(0, len(research_answer), 24):
                yield research_answer[start:start + 24]
            return
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

        checkpoint = get_checkpoint(canonical_model_id)
        answer = "".join(runtime.stream(
            checkpoint=checkpoint,
            messages=final_messages,
            max_new_tokens=min(
                max(1, max_new_tokens),
                int(model.get("max_new_tokens", 512)),
            ),
            temperature=float(model.get("temperature", 0.7)),
        ))
        if not tool_results and not is_readable_response(answer):
            yield UNREADABLE_FALLBACK
            return
        for start in range(0, len(answer), 24):
            yield answer[start:start + 24]



engine = NovyrixEngine()
