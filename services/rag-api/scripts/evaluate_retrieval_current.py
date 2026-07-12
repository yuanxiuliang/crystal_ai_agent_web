from __future__ import annotations

import json
from pathlib import Path
import re

from src.config import PROJECT_ROOT
from src.retrieval.milvus_hybrid import MilvusHybridRetriever


QUERIES = [
    {
        "qid": "q1_zh_formula_cvt_temp",
        "query": "ZnIn2S4 的 CVT 生长温度是多少？",
        "expect_formula": "ZnIn2S4",
        "expect_text": ["900 C", "850 C"],
    },
    {
        "qid": "q2_en_formula_cvt_temp",
        "query": "ZnIn2S4 chemical vapor transport source-zone and crystal-growth-zone temperature",
        "expect_formula": "ZnIn2S4",
        "expect_text": ["900 C", "850 C"],
    },
    {
        "qid": "q3_kcl_flux",
        "query": "CaTaO2N 使用 KCl 助熔剂的生长条件是什么？",
        "expect_formula": "CaTaO2N",
        "expect_text": ["KCl", "950 C"],
    },
    {
        "qid": "q4_biocl_hcl_cvt",
        "query": "BiOCl 的 CVT 源区和晶体区温度是多少？",
        "expect_formula": "BiOCl",
        "expect_text": ["699.85 C", "619.85 C"],
    },
    {
        "qid": "q5_lipc_ga_flux",
        "query": "Which LiB12PC record used Ga flux and was cooled from 1500 C to 1200 C?",
        "expect_formula": "LiB12PC",
        "expect_text": ["Ga", "1500 C", "1200 C"],
    },
    {
        "qid": "q6_semantic_no_formula_cvt_i2",
        "query": (
            "single crystals grown by chemical vapor transport using I2 with source-zone "
            "900 C and crystal-growth-zone 850 C"
        ),
        "expect_formula": "ZnIn2S4",
        "expect_text": ["I2", "900 C", "850 C"],
    },
    {
        "qid": "q7_semantic_dual_role",
        "query": (
            "flux growth record where selenium has ratio 0.1 and served as both starting "
            "material and additive"
        ),
        "expect_formula": "Ba2BiFeSe5",
        "expect_text": ["Se with ratio 0.1", "served as both"],
    },
    {
        "qid": "q8_semantic_hcl_no_formula",
        "query": (
            "chemical vapor transport record using HCl with source-zone temperature about "
            "700 C and crystal-growth-zone about 620 C"
        ),
        "expect_formula": "BiOCl",
        "expect_text": ["HCl", "699.85 C", "619.85 C"],
    },
]


def main() -> int:
    retriever = MilvusHybridRetriever()
    print(f"collection={retriever.config.milvus_collection}")
    print(f"model={retriever.config.embedding_model}")
    print(f"dim={retriever.config.embedding_dim}")

    for mode in ["dense", "bm25", "hybrid"]:
        print(f"\n=== {mode.upper()} ===")
        hits_at_1 = hits_at_3 = hits_at_5 = text_hits_at_5 = 0
        for item in QUERIES:
            hits = retriever.search(item["query"], mode=mode, top_k=5, trace=False)
            formulas = [hit.get("material_formula") for hit in hits]
            texts = [hit.get("source_text", "") for hit in hits]
            expected_formula = item["expect_formula"]
            ranks = [index + 1 for index, formula in enumerate(formulas) if formula == expected_formula]
            rank = ranks[0] if ranks else None
            if rank == 1:
                hits_at_1 += 1
            if rank and rank <= 3:
                hits_at_3 += 1
            if rank and rank <= 5:
                hits_at_5 += 1
            text_ok = any(all(part in text for part in item["expect_text"]) for text in texts)
            if text_ok:
                text_hits_at_5 += 1
            top = hits[0] if hits else {}
            print(
                f"{item['qid']}: rank_formula={rank} text_hit@5={text_ok} "
                f"top={top.get('material_formula')} | {top.get('doi')} | "
                f"{float(top.get('score') or 0):.6f}"
            )
            print("  top_text=", (top.get("source_text") or "")[:220].replace("\n", " "))
        total = len(QUERIES)
        print(
            f"SUMMARY {mode}: formula hit@1={hits_at_1}/{total}, "
            f"hit@3={hits_at_3}/{total}, hit@5={hits_at_5}/{total}, "
            f"required-text hit@5={text_hits_at_5}/{total}"
        )
    evaluate_formula_sample(retriever)
    return 0


def evaluate_formula_sample(retriever: MilvusHybridRetriever) -> None:
    path = PROJECT_ROOT / "data" / "processed" / "growth_records.text_only.jsonl"
    if not path.exists():
        return
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number % 157 != 1:
                continue
            text = json.loads(line)["text"]
            formula = _extract_formula(text)
            if formula:
                rows.append((formula, text))
            if len(rows) >= 40:
                break

    print("\n=== FORMULA SAMPLE TOP1 ===")
    for mode in ["dense", "bm25", "hybrid"]:
        hits = 0
        misses = []
        for formula, text in rows:
            query = f"{formula} single crystal growth conditions"
            results = retriever.search(query, mode=mode, top_k=1, trace=False)
            top_formula = results[0].get("material_formula") if results else None
            if top_formula == formula:
                hits += 1
            else:
                misses.append((formula, top_formula))
        print(f"{mode}: top1 formula hit={hits}/{len(rows)}")
        if misses:
            print("  sample_misses=", misses[:5])


def _extract_formula(text: str) -> str | None:
    match = re.search(r"\bFor\s+(.+?)\s+single crystals\b", text)
    return match.group(1).strip() if match else None


if __name__ == "__main__":
    raise SystemExit(main())
