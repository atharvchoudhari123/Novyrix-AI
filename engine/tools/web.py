from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _DuckDuckGoParser(HTMLParser):
    """Extract readable search-result titles, URLs, and snippets."""

    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture = None
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())

        if "result" in classes and self._current is None:
            self._current = {"title": "", "url": "", "snippet": ""}

        if self._current is None:
            return

        if tag == "a" and "result__a" in classes:
            self._capture = "title"
            self._buffer = []
            self._current["url"] = attrs.get("href", "")
        elif "result__snippet" in classes:
            self._capture = "snippet"
            self._buffer = []

    def handle_endtag(self, tag):
        if self._current is None or self._capture is None:
            return

        if tag in {"a", "div", "span"}:
            text = " ".join("".join(self._buffer).split())
            if text:
                self._current[self._capture] = text
            self._capture = None
            self._buffer = []

    def handle_data(self, data):
        if self._current is not None and self._capture:
            self._buffer.append(data)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self._current is not None and tag == "div":
            # DuckDuckGo closes each result in a result__body/result container.
            if self._current.get("title"):
                self.results.append(self._current)
                self._current = None
                self._capture = None
                self._buffer = []
                return
        if self._current is None or self._capture is None:
            return
        if tag in {"a", "span"}:
            text = " ".join("".join(self._buffer).split())
            if text:
                self._current[self._capture] = text
            self._capture = None
            self._buffer = []


def _clean_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    if parsed.path.startswith("/l/"):
        params = urllib.parse.parse_qs(parsed.query)
        url = params.get("uddg", [url])[0]
    return html.unescape(url)


def _fallback_text(text: str, max_chars: int) -> str:
    text = re.sub(r"<script\\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return " ".join(text.split())[:max_chars]


def search_web(query: str, max_results: int = 6, max_chars: int = 12000):
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query is empty.")

    # DuckDuckGo's HTML endpoint is intentionally used instead of returning
    # an entire search-engine page. Small local models need concise evidence,
    # not thousands of lines of HTML.
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) Novyrix/1.0",
            "Accept": "text/html",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return {
            "query": query,
            "source": "duckduckgo",
            "results": [],
            "error": f"Web search failed: {exc}",
            "verified": False,
        }

    parser = _DuckDuckGoParser()
    parser.feed(text)

    results = []
    seen = set()
    for item in parser.results:
        title = " ".join(item.get("title", "").split())
        snippet = " ".join(item.get("snippet", "").split())
        result_url = _clean_url(item.get("url", ""))
        key = (title, result_url)
        if not title or key in seen:
            continue
        seen.add(key)
        results.append({
            "title": title,
            "url": result_url,
            "snippet": snippet,
        })
        if len(results) >= max_results:
            break

    # If the search provider changes its HTML, return a bounded fallback rather
    # than pretending that the raw page is a useful factual result.
    if not results:
        return {
            "query": query,
            "source": "duckduckgo",
            "results": [],
            "raw_fallback": _fallback_text(text, max_chars),
            "verified": False,
            "warning": "No structured search results were extracted; do not treat the fallback as verified facts.",
        }

    return {
        "query": query,
        "source": "duckduckgo",
        "results": results,
        "verified": True,
        "instruction": (
            "Use these search results as evidence. Only state facts supported by "
            "the snippets/URLs. If the results do not establish an answer, say "
            "that the search did not verify it instead of guessing."
        ),
    }


def fetch_webpage(url: str, max_chars: int = 30000):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed.")
    request = urllib.request.Request(url, headers={"User-Agent": "Novyrix/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        text = response.read().decode("utf-8", errors="ignore")
        return {
            "url": url,
            "content_type": response.headers.get("content-type", ""),
            "content": text[:max_chars],
        }
