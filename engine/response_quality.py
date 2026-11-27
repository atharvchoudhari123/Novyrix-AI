"""Small, conservative safeguards for obviously corrupted scratch-model output."""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path


_COMMON_WORDS = frozenset(
    "a about after again all also am an and any are as at be because been before "
    "being between both but by can could did do does doing down each even every "
    "few for from get give go good had has have he her here hers him his how i "
    "if in into is it its just know like may me more most much must my no not of "
    "on one only or other our out over people please said same say see she should "
    "so some such take than that the their them then there these they this those "
    "through to too under up us use very was we well were what when where which "
    "who why will with would yes you your hello hi hey thanks okay ok"
    .split()
)
_WORDS = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)*")
_REPEATED_CHARACTER = re.compile(r"([^\s])\1{3,}", re.IGNORECASE)
_REPEATED_CHUNK = re.compile(r"([a-z]{1,3})\1{2,}", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```([a-zA-Z0-9_+-]*)[^\S\n]*\n(.*?)```", re.DOTALL)
_BROKEN_PHRASES = re.compile(
    r"\bmy\s+to\b"
    r"|\bhere\s+are\s+the\s+(?:purpose|reason|answer|idea|result)\b"
    r"|\b(?:is|are|was|were)\s+survive\b"
    r"|\bin\s+a\s+complete\s+of\b"
    r"|\bis\s+a\s+lower\s+to\b"
    r"|\bto\s+the\s+looked\b",
    re.IGNORECASE,
)


def _word_stem(word: str) -> str:
    """A tiny suffix normalizer used only to detect obvious repetition loops."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word
UNREADABLE_FALLBACK = (
    "I couldn't produce a reliable answer, so I discarded the generated text. "
    "Try a simpler or more specific prompt; this model may need more training."
)


def is_simple_greeting(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:hi|hello|hey|good morning|good afternoon|good evening)[!.?\s]*",
            text or "",
            re.IGNORECASE,
        )
    )


def is_coding_capability_question(text: str) -> bool:
    """Answer simple coding-capability checks without sampling the weak LM."""
    value = (text or "").strip().lower()
    # Accommodate the common adjacent-key transposition in "can".
    value = re.sub(r"\bcna\b", "can", value)
    value = re.sub(r"\s+", " ", value)
    return bool(
        re.fullmatch(
            r"(?:can|could|will) you (?:(?:write |help me (?:write )?)?(?:some )?(?:code|coding|program|programming)|"
            r"(?:help me )?(?:debug|fix|troubleshoot) (?:my )?(?:python|javascript|typescript|java|c\+\+|c#|go|rust|code|program)(?: code)?)[?.! ]*",
            value,
        )
    )


@lru_cache(maxsize=1)
def _system_dictionary() -> frozenset[str]:
    """Use an installed OS English dictionary when available; never require it."""
    for candidate in (Path("/usr/share/dict/words"), Path("/usr/dict/words")):
        try:
            return frozenset(
                word.lower()
                for word in candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
                if word.isascii() and word.isalpha()
            )
        except OSError:
            continue
    return frozenset()


def is_readable_response(text: str) -> bool:
    """Catch clear token leakage and severe character-level model collapse.

    This intentionally errs toward accepting unusual words; it is a guard for
    blatant corruption, not a grammar checker or a measure of correctness.
    """
    value = (text or "").strip()
    if not value:
        return False
    if re.search(r"<\|(?:im_start|im_end|assistant|user|system)[^|]*\|>", value):
        return False
    if _REPEATED_CHARACTER.search(value) or _REPEATED_CHUNK.search(value):
        return False
    if _BROKEN_PHRASES.search(value):
        return False
    if value.count("```") % 2:
        return False

    code_like = bool(
        "```" in value
        or re.search(r"(?m)^\s*(?:def|class|import|from|return|if|for|while|const|function)\b", value)
        or re.search(r"=>|[{};]", value)
    )
    if code_like:
        fenced_blocks = _FENCED_CODE.findall(value)
        for language, source in fenced_blocks:
            if language.lower() in {"py", "python", "python3"}:
                try:
                    ast.parse(source)
                except SyntaxError:
                    return False
        code_content = re.sub(r"```[^\n]*\n?", "", value).strip()
        code_content = code_content.replace("```", "").strip()
        return bool(code_content)

    words = [word.lower() for word in _WORDS.findall(value)]
    if not words:
        return False
    if len(words) < 5:
        return all(
            word in _COMMON_WORDS
            or word.replace(",", "").replace(".", "").isdigit()
            or (word.isalpha() and len(word) >= 5)
            for word in words
        )

    common_ratio = sum(word in _COMMON_WORDS for word in words) / len(words)
    if common_ratio < 0.16:
        return False

    # Scratch models sometimes produce fluent-looking loops such as
    # "the spectral claim ... trained ... trained ... training". Stop words
    # can make that look readable by ratio alone, so check whether the actual
    # content keeps cycling through the same few word stems.
    content_stems = [
        _word_stem(word)
        for word in words
        if word not in _COMMON_WORDS and len(word) >= 4
    ]
    if len(content_stems) >= 6:
        repeated_content_ratio = 1 - len(set(content_stems)) / len(content_stems)
        if repeated_content_ratio >= 0.3:
            return False

    # Catch fluent-looking strings made mostly of invented words. Keep the
    # dictionary check limited to long prose so identifiers and code remain
    # valid; punctuation-heavy output is exempt from this prose heuristic.
    dictionary = _system_dictionary()
    content_words = [word for word in words if len(word) >= 4]
    prose_like = len(re.findall(r"[.!?,;:]", value)) < max(3, len(words) // 8)
    if len(content_words) >= 8 and dictionary and prose_like:
        known_ratio = sum(word in dictionary for word in content_words) / len(content_words)
        if known_ratio < 0.78:
            return False
    if len(content_words) >= 8 and prose_like:
        unusual_ratio = sum(
            len(word) >= 7 and not re.search(r"[aeiouy]", word)
            for word in content_words
        ) / len(content_words)
        if unusual_ratio >= 0.2:
            return False
    return True
