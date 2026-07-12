from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, settings
from ..ingestion.normalizer import normalize_record


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "growth_records.normalized.jsonl"
DEFAULT_TEXT_ONLY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "growth_records.text_only.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-normalize",
        description="Normalize raw single-crystal growth JSONL records into retrieval-ready text.",
    )
    parser.add_argument(
        "--input", default=settings.growth_records_path, help="Input raw JSONL path."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output normalized JSONL path."
    )
    parser.add_argument(
        "--output-format",
        choices=["full", "text-only"],
        default="full",
        help="Write full normalized records or text-only records with a single text field.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument(
        "--preview", type=int, default=3, help="Print the first N normalized records."
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write output.")
    return parser


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.output == str(DEFAULT_OUTPUT) and args.output_format == "text-only":
        output_path = DEFAULT_TEXT_ONLY_OUTPUT

    records: list[dict[str, Any]] = []
    method_counter: Counter[str] = Counter()
    missing_conditions = 0

    for index, (_, raw) in enumerate(iter_jsonl(input_path), start=1):
        if args.limit is not None and index > args.limit:
            break
        normalized = normalize_record(raw)
        if args.output_format == "text-only":
            payload = {"text": normalized.normalized_text}
        else:
            payload = normalized.to_json_dict()
        records.append(payload)
        method_counter[normalized.method_normalized] += 1
        if not normalized.growth:
            missing_conditions += 1

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"input={input_path}")
    print(f"output={output_path if not args.dry_run else '(dry-run)'}")
    print(f"output_format={args.output_format}")
    print(f"processed={len(records)}")
    print(f"methods={dict(method_counter)}")
    print(f"missing_growth_conditions={missing_conditions}")

    preview_count = min(args.preview, len(records))
    if preview_count:
        print("\nPreview")
        for index, record in enumerate(records[:preview_count], start=1):
            print(f"\n[{index}]")
            if args.output_format == "text-only":
                print(json.dumps(record, ensure_ascii=False))
                continue
            metadata = record["metadata"]
            print(f"{metadata['record_id']}")
            print(f"formula={metadata['formula']}")
            print(f"method={metadata['growth_method_normalized']}")
            print(f"metadata={json.dumps(metadata, ensure_ascii=False, sort_keys=True)}")
            print(record["normalized_text"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
