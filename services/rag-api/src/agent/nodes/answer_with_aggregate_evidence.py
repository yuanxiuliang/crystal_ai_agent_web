from __future__ import annotations

from ..state import GrowthRAGState
from ..utils import error, trace


async def answer_with_aggregate_evidence(state: GrowthRAGState) -> dict:
    result = state["aggregate_result"]
    if result is None:
        return {
            "errors": [
                error(
                    "answer_with_aggregate_evidence",
                    "missing_aggregate_result",
                    "Aggregate result is missing.",
                    False,
                )
            ]
        }
    query = result["query"]
    lines = [
        "## 真实记录统计",
        "",
        "### 检索条件",
        f"- **{query['label']}**",
        (
            f"- 严格匹配到 **{result['total_records']} 条记录**、"
            f"**{result['total_formulas']} 个化学式**、"
            f"**{result['total_dois']} 个 DOI**。"
        ),
        "",
    ]
    if query["kind"] == "element_method_distribution":
        lines.extend(
            [
                "### 方法分布",
                "| 生长方法 | 记录数 | 化学式数 | DOI 数 |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {group['label']} | {group['record_count']} | "
                    f"{group['formula_count']} | {group['doi_count']} |"
                    for group in result["groups"]
                ],
            ]
        )
    else:
        heading = "匹配的已报道材料"
        if query["kind"] == "transport_agent_material_catalog":
            heading = "使用该传输剂的已报道材料"
        elif query["kind"] == "reactant_product_catalog":
            heading = "使用该起始原料的已报道材料"
        lines.extend(
            [
                f"### {heading}",
                "| 化学式 | 方法 | 记录数 | DOI 数 |",
                "| --- | --- | ---: | ---: |",
                *[
                    f"| {group['label']} | {group['growth_method'] or '未提供'} | "
                    f"{group['record_count']} | {group['doi_count']} |"
                    for group in result["groups"]
                ],
            ]
        )
    lines.extend(
        [
            "",
            "### 证据边界",
            "以上是当前知识库中满足严格结构化条件的真实记录统计。"
            "记录数反映语料覆盖和已收录路线数量，不代表实验成功率、最优条件或完整文献比例。",
            "下方列出可追溯的代表性记录；完整来源可在证据面板查看。",
        ]
    )
    answer = "\n".join(lines)
    return {
        "final_answer": answer,
        "selected_evidence_kind": "literature_record",
        "trace": [
            trace(
                "answer_with_aggregate_evidence",
                "answered",
                {"kind": query["kind"], "citation_count": len(state["citations"])},
            )
        ],
    }
