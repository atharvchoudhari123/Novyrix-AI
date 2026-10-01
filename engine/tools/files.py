from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _resolve(path: str) -> Path:
    target = (ROOT / str(path)).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("Path escapes the LumaCore project root.") from exc
    return target

def list_files(path: str = ".", recursive: bool = True):
    base = _resolve(path)
    iterator = base.rglob("*") if recursive else base.glob("*")
    return [
        str(p.relative_to(ROOT))
        for p in iterator
        if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts
    ][:500]

def read_file(path: str, max_chars: int = 50000):
    return _resolve(path).read_text(encoding="utf-8")[:max_chars]

def write_file(path: str, content: str, overwrite: bool = True):
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(path)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size}

def edit_file(path: str, old: str, new: str):
    target = _resolve(path)
    content = target.read_text(encoding="utf-8")
    if old not in content:
        raise ValueError("The requested text was not found.")
    target.write_text(content.replace(old, new, 1), encoding="utf-8")
    return {"path": str(target.relative_to(ROOT)), "changed": True}

def search_files(query: str, path: str = "."):
    base = _resolve(path)
    hits = []
    for p in base.rglob("*"):
        if not p.is_file() or ".git" in p.parts or ".venv" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if query.lower() in text.lower():
            hits.append(str(p.relative_to(ROOT)))
            if len(hits) >= 100:
                break
    return hits
