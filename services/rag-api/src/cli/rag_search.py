from __future__ import annotations

import argparse

from ..agent.state import RetrievalFilters
from ..retrieval.milvus_hybrid import MilvusHybridRetriever
from ..retrieval.query_expansion import expand_query_for_bm25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-search",
        description="Search Milvus growth records with BM25, dense, or hybrid retrieval.",
    )
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--formula", help="Optional exact formula filter, e.g. ZnIn2S4.")
    parser.add_argument("--method", help="Optional growth method filter, e.g. CVT or Flux.")
    parser.add_argument("--doi", help="Optional DOI filter.")
    parser.add_argument("--trace", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    retriever = MilvusHybridRetriever()
    filters: RetrievalFilters | None = None
    if args.formula or args.method or args.doi:
        filters = {
            "material_formula": args.formula,
            "material_name": None,
            "growth_method": args.method,
            "temperature_min": None,
            "temperature_max": None,
            "atmosphere": None,
            "doi": args.doi,
        }
    hits = retriever.search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        trace=args.trace,
        filters=filters,
    )
    print(f"QUERY: {args.query}")
    if filters:
        print(f"FILTERS: formula={args.formula} method={args.method} doi={args.doi}")
    if args.mode in {"bm25", "hybrid"}:
        print(f"BM25_QUERY: {expand_query_for_bm25(args.query)}")
    for index, hit in enumerate(hits, start=1):
        print(
            f"\n[{index}] score={hit['score']:.6f} "
            f"dense={hit['dense_score']} sparse={hit['sparse_score']}"
        )
        print(f"record_id={hit['record_id']}")
        print(f"formula={hit['material_formula']} method={hit['growth_method']} doi={hit['doi']}")
        if hit["temperature_program"]:
            print(f"temperature={hit['temperature_program']}")
        if args.trace and hit["matched_fields"]:
            print("trace=" + "; ".join(hit["matched_fields"]))
        print(hit["source_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
