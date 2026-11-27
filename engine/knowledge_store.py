"""Small local, source-linked cache of web research (not model training data)."""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


_STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it list me of on or tell that the this to what when where which who why with year".split()
)


class ResearchKnowledgeStore:
    """Persist short attributed search snippets for later, similar questions."""

    def __init__(self, path: str | None = None):
        configured = path or os.environ.get("NOVYRIX_KNOWLEDGE_PATH")
        self.path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "research_memory.json"
        self._lock = threading.RLock()

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term for term in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
            if term not in _STOP_WORDS
        }

    def _read(self) -> list[dict]:
        try:
            content = json.loads(self.path.read_text(encoding="utf-8"))
            return content if isinstance(content, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(records[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def remember(self, query: str, result: dict) -> bool:
        """Cache only structured results; never cache raw pages or failed searches."""
        if not isinstance(result, dict) or result.get("verified") is not True:
            return False
        entries = []
        for item in result.get("results", [])[:4]:
            if not isinstance(item, dict) or not item.get("title") or not item.get("url"):
                continue
            entries.append({
                "title": str(item["title"])[:240],
                "url": str(item["url"])[:1200],
                "snippet": str(item.get("snippet", ""))[:400],
            })
        if not entries:
            return False
        record = {
            "query": str(query)[:500],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "results": entries,
        }
        with self._lock:
            records = self._read()
            qterms = self._terms(record["query"])
            records = [r for r in records if self._terms(str(r.get("query", ""))) != qterms]
            records.append(record)
            self._write(records)
        return True

    def find(self, query: str, max_age_days: int = 30) -> dict | None:
        terms = self._terms(query)
        if not terms:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        best = None
        best_score = 0.0
        with self._lock:
            records = self._read()
        for record in records:
            try:
                saved = datetime.fromisoformat(record["saved_at"])
                if saved.tzinfo is None:
                    saved = saved.replace(tzinfo=timezone.utc)
                if saved < cutoff:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            old_terms = self._terms(str(record.get("query", "")))
            score = len(terms & old_terms) / len(terms | old_terms) if old_terms else 0.0
            if score > best_score:
                best, best_score = record, score
        # Avoid using a vaguely related answer as if it answered this question.
        return best if best_score >= 0.6 else None


knowledge_store = ResearchKnowledgeStore()


def format_research_result(result: dict, cached: bool = False) -> str:
    items = result.get("results", []) if isinstance(result, dict) else []
    if not items:
        error = result.get("error", "No usable search results were found.") if isinstance(result, dict) else "No usable search results were found."
        return f"I couldn't verify an answer from web search. {error}"
    heading = "Saved web research (" + str(result.get("saved_at", ""))[:10] + ")" if cached else "Web search results (snippets; check the sources for full context):"
    lines = [heading]
    for item in items[:4]:
        title = str(item.get("title", "Untitled source"))
        snippet = str(item.get("snippet", "")).strip()
        url = str(item.get("url", ""))
        lines.append(f"- {title}: {snippet}\n  Source: {url}")
    return "\n".join(lines)
