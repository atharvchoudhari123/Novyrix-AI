from __future__ import annotations

import urllib.parse
import urllib.request

def search_web(query: str, max_chars: int = 20000):
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query is empty.")
    url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "LumaCore/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        text = response.read().decode("utf-8", errors="ignore")
    return {"query": query, "source": "google", "content": text[:max_chars]}

def fetch_webpage(url: str, max_chars: int = 30000):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed.")
    request = urllib.request.Request(url, headers={"User-Agent": "LumaCore/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        text = response.read().decode("utf-8", errors="ignore")
        return {
            "url": url,
            "content_type": response.headers.get("content-type", ""),
            "content": text[:max_chars],
        }
