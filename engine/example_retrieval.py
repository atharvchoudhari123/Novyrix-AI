"""Retrieve reviewed local examples for narrow tasks the scratch models miss."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_FILES = {
    "novyrix-4.0": ROOT / "training" / "manual_examples" / "novyrix_4_0.jsonl",
    "novyrix-5.0": ROOT / "training" / "manual_examples" / "novyrix_5_0.jsonl",
    "novyrix-6.0": ROOT / "training" / "manual_examples" / "novyrix_6_0.jsonl",
}


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_+#.]+", (text or "").lower()))


@lru_cache(maxsize=3)
def _load_examples(model_id: str) -> tuple[dict, ...]:
    path = _EXAMPLE_FILES.get(model_id)
    if path is None or not path.is_file():
        return ()

    examples = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            prompt = next(
                (m.get("content", "") for m in messages
                 if isinstance(m, dict) and m.get("role") == "user"),
                "",
            )
            answer = next(
                (m.get("content", "") for m in reversed(messages)
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                "",
            )
            groups = record.get("retrieval_groups", [])
            if prompt and answer and isinstance(groups, list):
                examples.append({
                    "prompt": _normalize(prompt),
                    "answer": answer.strip(),
                    "groups": tuple(
                        tuple(_normalize(str(term)) for term in group if str(term).strip())
                        for group in groups
                        if isinstance(group, list) and group
                    ),
                })
    return tuple(examples)


def retrieve_reviewed_answer(
    model_id: str, user_text: str, files=None, mode: str = "chat"
) -> str | None:
    """Return an answer only for an exact or explicitly tagged example topic."""
    if files or mode not in {"chat", "code", "debug", "api"}:
        return None
    query = _normalize(user_text)
    if not query:
        return None
    if re.search(
        r"\b(?:search|research|online|latest|current|newest|documentation|docs)\b",
        query,
    ):
        return None

    for example in _load_examples(model_id):
        if query == example["prompt"]:
            return example["answer"]
        groups = example["groups"]
        if groups and all(
            any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query) for term in group)
            for group in groups
        ):
            return example["answer"]
    return None
