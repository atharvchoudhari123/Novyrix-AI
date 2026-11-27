"""Export saved web research as reviewable chat examples for a Novyrix tier.

The default output is a candidate file and is not picked up by training. Pass
--approved only after checking the source rights and the generated examples.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.knowledge_store import ResearchKnowledgeStore  # noqa: E402

ALLOWED_LICENSES = {"CC0", "CC-BY", "public-domain"}


def build_example(record: dict, license_name: str) -> dict | None:
    query = " ".join(str(record.get("query", "")).split())
    results = record.get("results")
    if not query or not isinstance(results, list):
        return None

    sources = []
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title", "")).split())[:240]
        snippet = " ".join(str(item.get("snippet", "")).split())[:400]
        url = str(item.get("url", "")).strip()
        if not title or len(snippet) < 30 or not re.match(r"^https?://", url):
            continue
        key = (title.casefold(), url)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"title": title, "snippet": snippet, "url": url})
        if len(sources) == 4:
            break
    if not sources:
        return None

    saved_at = str(record.get("saved_at", ""))[:10]
    answer_lines = [
        "I found these source excerpts. They are search snippets, so check the linked pages for full context:"
    ]
    for source in sources:
        answer_lines.append(f"- {source['title']}: {source['snippet']}\n  Source: {source['url']}")
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "\n".join(answer_lines)},
        ],
        "source": {
            "dataset": "Novyrix local web research cache",
            "license": license_name,
            "saved_at": saved_at,
            "source_urls": [source["url"] for source in sources],
            "note": "Generated from search snippets; verify answer quality and source rights before training.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("4.0", "5.0", "6.0"), required=True)
    parser.add_argument("--input", type=Path, help="Candidate JSONL to approve after review (defaults to this tier's candidate export).")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--approved", action="store_true",
        help="Mark the export as training input after you have checked every source's rights and content.",
    )
    parser.add_argument(
        "--license", choices=sorted(ALLOWED_LICENSES),
        help="Required with --approved; use only when every included source allows this use.",
    )
    args = parser.parse_args()
    if args.approved and not args.license:
        parser.error("--approved requires --license, and every source must allow that license/use")
    if not args.approved and args.license:
        parser.error("--license is only accepted with --approved")

    tier_slug = args.tier.replace(".", "_")
    default_dir = ROOT / "training" / "data" / "research_cache"
    if args.approved:
        input_path = args.input or default_dir / f"novyrix_{tier_slug}_candidates.jsonl"
        if not input_path.is_file():
            parser.error(f"candidate file does not exist: {input_path}")
        examples = []
        with input_path.open("r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as error:
                    parser.error(f"invalid JSON at {input_path}:{line_no}: {error}")
                messages = example.get("messages") if isinstance(example, dict) else None
                if not isinstance(messages, list) or len(messages) < 2 or any(
                    not isinstance(message, dict)
                    or message.get("role") not in {"user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                    for message in messages
                ):
                    parser.error(f"invalid chat messages at {input_path}:{line_no}")
                source_info = example.get("source")
                if not isinstance(source_info, dict) or not source_info.get("source_urls"):
                    parser.error(f"missing source metadata at {input_path}:{line_no}")
                source_info["license"] = args.license
                example["source"] = source_info
                examples.append(example)
        if not examples:
            parser.error(f"no usable examples found in {input_path}")
        raw_count = len(examples)
    else:
        store = ResearchKnowledgeStore()
        records = store._read()
        examples = []
        seen = set()
        for record in records:
            example = build_example(record, "REVIEW_REQUIRED")
            if example is None:
                continue
            key = " ".join(example["messages"][0]["content"].casefold().split())
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
        raw_count = len(records)

    suffix = "approved" if args.approved else "candidates"
    output = args.output or default_dir / f"novyrix_{tier_slug}_{suffix}.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for example in examples:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(f"Exported {len(examples)} examples from {raw_count} {'reviewed candidates' if args.approved else 'cached records'} to {output}")
    if not args.approved:
        print("Candidate export only. Review each answer and source license before preparing training input.")
    else:
        manifest = output.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({
            "tier": args.tier,
            "license": args.license,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "examples": len(examples),
            "reviewed_by_user": True,
            "note": "Approval is the user's declaration that all source rights and examples were checked.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Approval manifest: {manifest}")


if __name__ == "__main__":
    main()
