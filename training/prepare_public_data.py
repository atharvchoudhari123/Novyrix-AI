"""Fetch pinned, permissively licensed corpora and prepare tier-specific SFT JSONL."""
from __future__ import annotations

import ast
import hashlib
import gzip
import json
import os
import re
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"
EXTERNAL = DATA / "external"
PREPARED = DATA / "prepared" / "renamed"
MANUAL_EXAMPLES = ROOT / "training" / "manual_examples"

OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
PYTHON_REVISION = "0ed98ff2a76c5d133d8c157b814189a5a17ebd20"
OASST_REPO = "OpenAssistant/oasst1"
PYTHON_REPO = "flytech/python-codes-25k"

OASST_SOURCE = {
    "dataset": OASST_REPO,
    "revision": OASST_REVISION,
    "license": "Apache-2.0",
    "url": f"https://huggingface.co/datasets/{OASST_REPO}/tree/{OASST_REVISION}",
    "filter": "English, reviewed, non-deleted, rank-0 assistant turns; sensitive/toxic labels filtered.",
}
PYTHON_SOURCE = {
    "dataset": PYTHON_REPO,
    "revision": PYTHON_REVISION,
    "license": "MIT (as stated by the pinned dataset card)",
    "url": f"https://huggingface.co/datasets/{PYTHON_REPO}/tree/{PYTHON_REVISION}",
    "filter": "Unique instruction/output pairs; Python output must parse with ast.parse. Syntax validation is not correctness verification.",
}
MANUAL_SOURCE = {
    "dataset": "Novyrix reviewed evaluation corrections",
    "revision": "project source",
    "license": "Original project content",
    "filter": "Targeted, reviewed answers written to correct observed tier-specific failures.",
}


def _download(repo: str, revision: str, filename: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        # curl uses the operating system trust store on macOS; TLS verification
        # remains enabled (unlike the Python urllib path in some local setups).
        subprocess.run(
            ["curl", "-fL", "--retry", "3", "--max-time", "600", url, "-o", str(temporary)],
            check=True,
        )
        if not temporary.stat().st_size:
            raise RuntimeError(f"Downloaded an empty dataset file from {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_oasst_message(item: dict) -> bool:
    if item.get("lang") != "en" or item.get("deleted") or item.get("review_result") is not True:
        return False
    if item.get("review_count", 0) < 2:
        return False
    labels = item.get("labels") or {}
    for name in ("spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content"):
        value = (labels.get(name) or {}).get("value")
        if value is not None and value >= 0.5:
            return False
    for name in ("toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"):
        value = (item.get("detoxify") or {}).get(name)
        if value is not None and value >= 0.8:
            return False
    return bool(str(item.get("text", "")).strip())


def prepare_oasst(path: Path) -> tuple[list[dict], int]:
    messages = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if _safe_oasst_message(item):
                messages[item["message_id"]] = item

    examples = []
    seen = set()
    for target in messages.values():
        if target.get("role") != "assistant" or target.get("rank") not in (None, 0):
            continue
        path_messages = []
        cursor = target
        valid = True
        while cursor:
            role = "assistant" if cursor.get("role") == "assistant" else "user"
            path_messages.append({"role": role, "content": str(cursor.get("text", "")).strip()})
            parent_id = cursor.get("parent_id")
            cursor = messages.get(parent_id) if parent_id else None
            if parent_id and cursor is None:
                valid = False
                break
        if not valid:
            continue
        path_messages.reverse()
        if not path_messages or path_messages[0]["role"] != "user":
            continue
        if any(a["role"] == b["role"] for a, b in zip(path_messages, path_messages[1:])):
            continue
        key = json.dumps(path_messages, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        examples.append({"messages": path_messages, "source": OASST_SOURCE})
    return examples, len(messages)


_PYTHON_BLOCK = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _valid_python_output(output: str) -> bool:
    blocks = _PYTHON_BLOCK.findall(output or "")
    candidates = blocks or [output]
    try:
        for code in candidates:
            if not code.strip():
                return False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(code)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def prepare_python(path: Path) -> tuple[list[dict], dict[str, int]]:
    examples = []
    seen = set()
    stats = {"rows": 0, "kept": 0, "invalid_python": 0, "duplicates": 0}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_python"] += 1
                continue
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if not instruction or not output or not _valid_python_output(output):
                stats["invalid_python"] += 1
                continue
            key = " ".join(instruction.casefold().split())
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            context = str(row.get("input", "")).strip()
            prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ],
                "source": PYTHON_SOURCE,
            })
            stats["kept"] += 1
    return examples, stats


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {error}") from error
    return records


def _read_manual_examples(tier: str) -> list[dict]:
    path = MANUAL_EXAMPLES / f"novyrix_{tier}.jsonl"
    if not path.is_file():
        return []
    records = _read_jsonl(path)
    for number, record in enumerate(records, 1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            for message in messages
        ) or not any(
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"Manual example needs user and assistant messages at {path}:{number}")
        record.setdefault("source", MANUAL_SOURCE)
    return records


def _write_jsonl_atomic(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    oasst_path = EXTERNAL / "oasst1" / "ready.messages.jsonl.gz"
    python_path = EXTERNAL / "python-codes-25k" / "python-codes-25k.jsonl"
    _download(OASST_REPO, OASST_REVISION, "2023-04-12_oasst_ready.messages.jsonl.gz", oasst_path)
    _download(PYTHON_REPO, PYTHON_REVISION, "python-codes-25k.jsonl", python_path)

    general, accepted_oasst_messages = prepare_oasst(oasst_path)
    coding, code_stats = prepare_python(python_path)
    if not general:
        raise RuntimeError("The quality filters produced no usable OASST1 conversation examples.")
    if not coding:
        raise RuntimeError("Python code validation produced no usable coding examples.")

    # Reuse prior curated corpora by capacity level. The old source filenames
    # are read-only here; newly prepared corpora use the renamed tier IDs.
    existing = {}
    for old_tier in ("3_2", "4_0", "5_7"):
        source_path = DATA / f"novyrix_{old_tier}.jsonl"
        existing[old_tier] = _read_jsonl(source_path) if source_path.is_file() else []

    manual_examples = {
        tier: _read_manual_examples(tier) for tier in ("4_0", "5_0", "6_0")
    }
    tier_records = {
        "4_0": general + coding + existing["3_2"] + manual_examples["4_0"],
        "5_0": general + coding + existing["4_0"] + manual_examples["5_0"],
        "6_0": general + coding + existing["5_7"] + manual_examples["6_0"],
    }
    PREPARED.mkdir(parents=True, exist_ok=True)
    for tier, records in tier_records.items():
        _write_jsonl_atomic(PREPARED / f"novyrix_{tier}.jsonl", records)

    source_manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [OASST_SOURCE, PYTHON_SOURCE, MANUAL_SOURCE],
        "downloaded_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (oasst_path, python_path)
        ],
        "filters_and_counts": {
            "oasst_accepted_messages": accepted_oasst_messages,
            "oasst_examples": len(general),
            "python": code_stats,
            "manual_examples_by_tier": {
                tier.replace("_", "."): len(records)
                for tier, records in manual_examples.items()
            },
        },
        "tier_mixtures": {
            "4.0": [OASST_REPO, PYTHON_REPO, "Novyrix 4.0 curated examples", "Novyrix reviewed evaluation corrections"],
            "5.0": [OASST_REPO, PYTHON_REPO, "Novyrix 5.0 curated examples", "Novyrix reviewed evaluation corrections"],
            "6.0": [OASST_REPO, PYTHON_REPO, "Novyrix 6.0 curated examples", "Novyrix reviewed evaluation corrections"],
        },
        "note": "All model parameters are initialized and trained by Novyrix; these external corpora provide examples only.",
    }
    (PREPARED / "SOURCES.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tier_sources = {
        "4_0": [OASST_SOURCE, PYTHON_SOURCE, MANUAL_SOURCE],
        "5_0": [OASST_SOURCE, PYTHON_SOURCE, MANUAL_SOURCE],
        "6_0": [OASST_SOURCE, PYTHON_SOURCE, MANUAL_SOURCE],
    }
    for tier, sources in tier_sources.items():
        specific = {
            **source_manifest,
            "sources": sources,
            "tier_mixture": source_manifest["tier_mixtures"][tier.replace("_", ".")],
        }
        (PREPARED / f"SOURCES_{tier}.json").write_text(
            json.dumps(specific, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for tier, records in tier_records.items():
        print(f"Novyrix {tier.replace('_', '.')}: {len(records)} records -> {PREPARED / f'novyrix_{tier}.jsonl'}")
    print(f"OASST messages={accepted_oasst_messages}; examples={len(general)}")
    print(f"Python rows={code_stats['rows']}; valid={code_stats['kept']}; rejected={code_stats['invalid_python']}; duplicates={code_stats['duplicates']}")
    print(f"Source/license manifest: {PREPARED / 'SOURCES.json'}")


if __name__ == "__main__":
    main()
