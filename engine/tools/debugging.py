from __future__ import annotations

import ast
import subprocess
import sys
from .files import ROOT, _resolve

def analyze_traceback(traceback_text: str):
    lines = [x.strip() for x in (traceback_text or "").splitlines() if x.strip()]
    return {
        "last_line": lines[-1] if lines else "",
        "lines": lines[-20:],
        "likely_location": next((x for x in reversed(lines) if 'File "' in x), None),
    }

def check_syntax(path: str):
    target = _resolve(path)
    if target.suffix != ".py":
        raise ValueError("check_syntax currently supports Python files.")
    ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    return {"ok": True, "path": str(target.relative_to(ROOT))}

def run_tests(path: str = "."):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }
