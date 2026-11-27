"""Stream license-checked public text into exact-size Novyrix web corpora.

Sources are the filtered Common Pile corpora whose records carry CC0 or public
domain metadata. This downloads only the portions needed for Novyrix's targets;
it does not scrape arbitrary sites or invent/pad examples.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "training" / "data" / "prepared" / "renamed" / "web"
MAX_TEXT_BYTES = int(os.environ.get("NOVYRIX_WEB_MAX_TEXT_BYTES", "2048"))

SOURCES = {
    "arxiv": {
        "repo": "common-pile/arxiv_abstracts_filtered",
        "url": "https://huggingface.co/datasets/common-pile/arxiv_abstracts_filtered",
        "license_rule": "record metadata license must identify CC0 / Creative Commons Zero",
        "accept": lambda value: bool(re.search(r"\b(cc0|creative commons zero)\b", value, re.I)),
    },
    "caselaw": {
        "repo": "common-pile/caselaw_access_project",
        "url": "https://huggingface.co/datasets/common-pile/caselaw_access_project",
        "license_rule": "record metadata license must be Public Domain",
        "accept": lambda value: "public domain" in value.casefold(),
    },
    "pubmed": {
        "repo": "common-pile/pubmed_filtered",
        "url": "https://huggingface.co/datasets/common-pile/pubmed_filtered",
        "license_rule": "record metadata must be Public Domain, CC0, or attribution-only CC BY; excludes SA/NC/ND",
        "accept": lambda value: (
            "public domain" in value.casefold()
            or bool(re.search(r"\b(cc0|creative commons zero)\b", value, re.I))
            or (
                "creative commons" in value.casefold()
                and "attribution" in value.casefold()
                and not re.search(r"share.?alike|non.?commercial|no.?derivatives|\bby-sa\b|\bby-nc\b|\bby-nd\b", value, re.I)
            )
        ),
    },
    "usgpo": {
        "repo": "common-pile/usgpo_filtered",
        "url": "https://huggingface.co/datasets/common-pile/usgpo_filtered",
        "license_rule": "record metadata license must be Public Domain",
        "accept": lambda value: "public domain" in value.casefold(),
    },
}

SHARDS = {
    "arxiv_base": ("arxiv", 2_000_000, 0),
    "pubmed_base": ("pubmed", 2_000_000, 0),
    "pubmed_extra": ("pubmed", 1_000_000, 2_000_000),
    "caselaw_base": ("caselaw", 2_000_000, 0),
    "caselaw_extra": ("caselaw", 1_200_000, 2_000_000),
    "usgpo": ("usgpo", 1_800_000, 0),
}

TIERS = {
    "4.0": ["arxiv_base"],
    "5.0": ["arxiv_base", "pubmed_base"],
    "6.0": ["arxiv_base", "pubmed_base", "pubmed_extra", "caselaw_base", "caselaw_extra", "usgpo"],
}


def _license_and_url(record: dict) -> tuple[str, str]:
    metadata = record.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    metadata = metadata if isinstance(metadata, dict) else {}
    license_name = str(metadata.get("license", record.get("license", ""))).strip()
    url = str(metadata.get("url", record.get("url", ""))).strip()
    return license_name, url


def _clean_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_TEXT_BYTES:
        text = encoded[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore").rstrip()
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_shard(name: str, source_key: str, count: int, skip: int, revision: str) -> dict:
    source = SOURCES[source_key]
    path = OUTPUT / f"{name}.jsonl"
    if path.is_file():
        with path.open("rb") as existing_file:
            existing = sum(1 for _ in existing_file)
        if existing == count:
            return {"path": str(path.relative_to(ROOT)), "examples": count, "sha256": _sha256_file(path), "reused": True}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(source["repo"], split="train", streaming=True, revision=revision)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT, delete=False)
    temporary = Path(handle.name)
    accepted = 0
    skipped = 0
    try:
        for row in dataset:
            text = _clean_text(row.get("text", ""))
            license_name, url = _license_and_url(row)
            if not text or len(text) < 80 or not source["accept"](license_name):
                continue
            if skipped < skip:
                skipped += 1
                continue
            item = {
                "id": str(row.get("id", "")),
                "text": text,
                "source": source["repo"],
                "source_url": url,
                "license": license_name,
            }
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            accepted += 1
            if accepted == count:
                break
        handle.close()
        if accepted != count:
            raise RuntimeError(
                f"{source['repo']} yielded only {accepted:,} records satisfying its "
                f"license/quality filters; {count:,} required. No target was padded."
            )
        os.replace(temporary, path)
    finally:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)

    return {"path": str(path.relative_to(ROOT)), "examples": count, "sha256": _sha256_file(path), "reused": False}


def main() -> None:
    if MAX_TEXT_BYTES < 256:
        raise ValueError("NOVYRIX_WEB_MAX_TEXT_BYTES must be at least 256")

    api = HfApi()
    revisions = {}
    for key, source in SOURCES.items():
        info = api.dataset_info(source["repo"])
        revisions[key] = info.sha

    shards = {}
    for name, (source_key, count, skip) in SHARDS.items():
        shards[name] = {
            **_write_shard(name, source_key, count, skip, revisions[source_key]),
            "dataset": SOURCES[source_key]["repo"],
            "dataset_url": SOURCES[source_key]["url"],
            "revision": revisions[source_key],
            "license_filter": SOURCES[source_key]["license_rule"],
            "text_byte_limit": MAX_TEXT_BYTES,
        }

    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "stream exact records from filtered Common Pile datasets; verify each record's license metadata; no padding",
        "shards": shards,
        "tiers": {
            tier: {
                "shards": names,
                "shard_details": {name: shards[name] for name in names},
                "web_examples": sum(shards[name]["examples"] for name in names),
            }
            for tier, names in TIERS.items()
        },
    }
    manifest_path = OUTPUT / "SOURCES.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for tier, details in manifest["tiers"].items():
        tier_path = OUTPUT / f"SOURCES_{tier.replace('.', '_')}.json"
        tier_path.write_text(
            json.dumps({**details, "prepared_at_utc": manifest["prepared_at_utc"]}, indent=2),
            encoding="utf-8",
        )
        print(f"Novyrix {tier}: {details['web_examples']:,} verified web examples")
    print(f"License and revision manifest: {manifest_path}")


if __name__ == "__main__":
    main()
