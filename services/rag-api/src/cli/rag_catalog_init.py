from __future__ import annotations

import argparse

from ..config import settings
from ..retrieval.fact_catalog import get_default_fact_catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-catalog-init",
        description="Build the structured real-record catalog without embedding data.",
    )
    parser.add_argument("--input", default=settings.growth_records_path)
    args = parser.parse_args()
    result = get_default_fact_catalog().sync_from_jsonl(args.input)
    print(
        f"catalog_status={result.status} records={result.record_count} "
        f"source_hash={result.source_hash[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
