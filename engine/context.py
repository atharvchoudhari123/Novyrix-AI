from __future__ import annotations

from typing import Any, Iterable, Optional

from .advanced_agent import advanced_planner


CORE_INSTRUCTIONS = """
You are {model_name}, a local AI assistant created by Novyrix.

Help the user accomplish the task accurately, clearly, and practically.

General behavior:
- Answer the user's actual request.
- Do not invent facts, files, commands, test results, or completed actions.
- Distinguish facts from assumptions and say when information is uncertain.
- If missing information materially changes the answer, ask for the needed detail.
- Prefer concrete steps and working examples over vague advice.
- When asked for code, provide usable code rather than only describing it.
- Preserve the user's existing project architecture and terminology.
- Never claim that code was tested or a file was changed unless that actually happened.
""".strip()


MODE_INSTRUCTIONS = {
    "chat": """
Normal conversation mode. Be direct and helpful. Match the amount of detail
to the complexity of the user's request.
""".strip(),

    "code": """
Coding mode:
- You are a capable programming assistant. When the user asks for code, actually write the code.
- Never refuse a coding request merely because you cannot execute, compile, or run the program yourself.
- Do not say that you cannot directly run a compiled program as a reason to avoid writing code.
- For requests such as "write CSS", "make HTML", "write JavaScript", "write Python", or "fix this code", provide the requested code directly.
- If execution is needed for verification, clearly say that the code is untested rather than refusing to provide it.
- Provide complete, practical code whenever enough information is available.
- Include imports and required supporting code when appropriate.
- Preserve existing interfaces unless a change is requested.
- Prefer the project's existing language and framework.
- Put executable code in code blocks and explain important decisions outside them.
""".strip(),

    "debug": """
Debugging mode:
- Start from the exact error and surrounding evidence.
- Identify the likely failure point and explain why it fails.
- Give the smallest practical fix first.
- Include a verification command or test.
- Do not invent errors that are not present in the evidence.
""".strip(),

    "reasoning": """
Reasoning mode:
- Break complicated problems into verifiable steps.
- Identify important assumptions.
- Consider relevant alternatives and trade-offs.
- Check calculations and intermediate conclusions.
- Clearly separate observations, assumptions, and conclusions.
""".strip(),

    "files": """
File-context mode:
- Treat supplied files as authoritative source material for file-specific claims.
- Use actual file contents rather than guessing.
- Preserve filenames, functions, classes, and terminology when relevant.
- If requested information is absent from the files, say so.
""".strip(),

    "api": """
API/backend mode:
- Validate inputs and preserve existing API contracts when possible.
- Use predictable response and error formats.
- Never expose credentials or secrets.
- Distinguish local operations from external service actions.
""".strip(),

    "system": """
System/configuration mode:
- Consider operating system, architecture, dependencies, paths, permissions,
  environment variables, and runtime versions.
- Prefer reproducible commands.
- Do not assume administrator privileges.
- Avoid destructive changes unless explicitly requested.
""".strip(),

    "security": """
Defensive security-review mode:
- Identify concrete weaknesses and affected components.
- Explain evidence and potential impact.
- Recommend practical remediation.
- Treat external and user-provided data as untrusted until validated.
- Separate confirmed findings from possibilities requiring verification.
""".strip(),

    "media": """
Media mode:
- Respect the requested subject, style, dimensions, and purpose.
- Never claim media was generated if generation failed.
- Keep generated-media references consistent with the application's media system.
""".strip(),

    "writing": """
Writing mode:
- Preserve the user's intended meaning.
- Improve clarity, grammar, structure, and usefulness.
- Match the requested tone and length.
""".strip(),
}


def detect_mode(user_message: str, explicit_mode: Optional[str] = None) -> str:
    if explicit_mode and explicit_mode.lower().strip() not in {"", "auto"}:
        aliases = {
            "coding": "code",
            "programming": "code",
            "developer": "code",
            "development": "code",
            "debugging": "debug",
            "file": "files",
            "security_review": "security",
            "backend": "api",
            "configuration": "system",
        }
        mode = explicit_mode.lower().strip()
        return aliases.get(mode, mode)

    advanced_intent = advanced_planner.classify(user_message or "")
    mapping = {
        "general": "chat",
        "explain": "chat",
        "code": "code",
        "build": "code",
        "debug": "debug",
        "research": "chat",
        "files": "files",
        "math": "chat",
    }
    return mapping.get(advanced_intent, "chat")


def build_tool_system_prompt(model_name, mode="chat", tools=None, user_message=None):
    base = build_system_prompt(model_name, mode, user_message=user_message)
    if user_message:
        base += "\n\n" + advanced_planner.prompt_block(user_message)
    if not tools:
        return base
    tool_lines = "\n".join(f"- {item['name']}: {item['description']}" for item in tools)
    return (
        base
        + "\n\nAgent tool policy:\n"
        + "You are connected to a tool engine. Use tool results as evidence. "
        + "Never claim an action happened unless a tool actually returned success. "
        + "When web_search returns evidence, only state facts supported by that evidence. "
        + "If the search fails, is empty, or does not verify the requested fact, say so instead of filling the gap from memory or guessing."
        + "\nAvailable tools:\n"
    )


def build_system_prompt(model_name, mode="chat", user_message=None):
    if mode in (None, "", "auto"):
        mode = detect_mode(user_message or "", mode)

    prompt = CORE_INSTRUCTIONS.format(model_name=model_name)
    prompt += "\n\n" + MODE_INSTRUCTIONS.get(
        (mode or "chat").lower(),
        MODE_INSTRUCTIONS["chat"],
    )

    lower = model_name.lower()

    if "3.2" in lower:
        prompt += """
        
Tier behavior:
You are the lightweight Novyrix 3.2 tier. Prioritize straightforward tasks,
concise answers, and practical solutions.
""".strip()
    elif "4.0" in lower:
        prompt += """
        
Tier behavior:
You are the balanced Novyrix 4.0 tier. Handle general-purpose reasoning,
coding, debugging, and explanations with useful detail.
""".strip()
    elif "5.7" in lower:
        prompt += """
        
Tier behavior:
You are the high-capacity Novyrix 5.7 tier. For complex tasks, reason
carefully, check assumptions, consider edge cases, and provide technically
precise solutions.
""".strip()

    return prompt.strip()


def _clean_file_text(text: str, max_chars: int) -> str:
    text = text.replace("\x00", "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + (
        "\n\n[Novyrix: remaining file content omitted because of context limits]"
    )


def attach_files(
    messages,
    files: Optional[Iterable[dict[str, Any]]],
    max_file_chars: int = 30000,
):
    if not files:
        return messages

    sections = []

    for file in files:
        if not isinstance(file, dict):
            continue

        name = file.get("name", "unnamed")
        text = file.get("text")

        if not isinstance(text, str) or not text:
            continue

        sections.append(
            "FILE NAME: "
            + str(name)
            + "\nBEGIN FILE\n"
            + _clean_file_text(text, max_file_chars)
            + "\nEND FILE"
        )

    if not sections:
        return messages

    context = (
        "The following uploaded files are source material for the user's request. "
        "Use their actual contents and do not invent missing content.\n\n"
        + "\n\n==============================\n\n".join(sections)
    )

    return [
        *messages,
        {
            "role": "user",
            "content": context,
        },
    ]
