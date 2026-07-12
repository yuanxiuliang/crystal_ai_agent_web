from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from ..streaming.events import StreamEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-chat",
        description="Validate the AgentWeb GrowthRAG workflow from the command line.",
    )
    parser.add_argument("message", nargs="?", help="Single user message. Omit it for interactive mode.")
    parser.add_argument("--user-id", default="cli-user", help="User id used by the graph state.")
    parser.add_argument("--session-id", default="cli-session", help="Session id used by the graph state.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieval records to request.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval mode passed to the graph.",
    )
    parser.add_argument("--force-retrieve", action="store_true", help="Force the retrieval path.")
    parser.add_argument("--trace", action="store_true", help="Print node, route, retrieval, evidence events.")
    parser.add_argument("--json", action="store_true", help="Print the final response as JSON.")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM for offline workflow checks.")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"

    if args.message:
        final = await run_turn(args, args.message)
        return 0 if final and not final.get("errors") else 1

    print(
        "AgentWeb RAG CLI. 输入 /exit 退出，/trace on 或 /trace off 切换节点追踪，"
        "/whoami 查看测试身份，/memory 查看当前用户长期记忆。"
    )
    while True:
        try:
            message = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return 0
        if message == "/trace on":
            args.trace = True
            print("trace=on")
            continue
        if message == "/trace off":
            args.trace = False
            print("trace=off")
            continue
        if message == "/whoami":
            print(f"user_id={args.user_id} session_id={args.session_id}")
            continue
        if message == "/memory":
            await print_long_memory(args.user_id)
            continue

        await run_turn(args, message)


async def run_turn(
    args: argparse.Namespace,
    message: str,
) -> dict[str, Any] | None:
    payload = {
        "user_id": args.user_id,
        "session_id": args.session_id,
        "message": message,
        "options": {
            "force_retrieve": args.force_retrieve,
            "top_k": args.top_k,
            "retrieval_mode": args.retrieval_mode,
            "stream_trace": args.trace,
        },
    }
    final: dict[str, Any] | None = None
    answer_chunks: list[str] = []

    try:
        from ..agent.graph import GrowthRAGGraph

        async for event in GrowthRAGGraph().stream(payload):
            if args.json:
                if event.event == "final":
                    final = event.data
                continue
            handle_event(event, args.trace, answer_chunks)
            if event.event == "final":
                final = event.data
    except Exception as exc:  # noqa: BLE001 - CLI should surface graph/runtime failures cleanly.
        print(f"\n[error] {exc}", file=sys.stderr)
        return None

    if args.json:
        print(json.dumps(final or {}, ensure_ascii=False, indent=2))
    elif final:
        print_final_metadata(final, printed_answer=bool(answer_chunks))
    return final


async def print_long_memory(user_id: str) -> None:
    from ..memory.store import get_memory_store

    store = get_memory_store()
    memories = await asyncio.to_thread(store.load_long_memories, user_id=user_id, query="")
    print(f"[memory] user_id={user_id} visible_items={len(memories)}")
    for item in memories:
        print(f"- {item['type']} | {item['content']}")


def handle_event(event: StreamEvent, trace_enabled: bool, answer_chunks: list[str]) -> None:
    data = event.data
    if event.event != "token" and trace_enabled:
        ensure_trace_line_start(answer_chunks)

    if event.event == "node_started" and trace_enabled:
        print(f"[node:start] {data.get('node')} | {data.get('label')}")
    elif event.event == "node_finished" and trace_enabled:
        print(f"[node:done] {data.get('node')} | {data.get('label')}")
    elif event.event == "route_decision" and trace_enabled:
        print(
            "[route] "
            f"{data.get('intent')} | should_retrieve={data.get('should_retrieve')} "
            f"| confidence={data.get('confidence')} | {data.get('reason')}"
        )
    elif event.event == "retrieval_plan" and trace_enabled:
        filters = data.get("filters", {})
        print(
            "[plan] "
            f"query={data.get('query_text')} | filters={json.dumps(filters, ensure_ascii=False)} "
            f"| top_k={data.get('top_k')}"
        )
    elif event.event == "retrieval_result" and trace_enabled:
        print(
            "[record] "
            f"{data.get('record_id')} | score={data.get('score')} "
            f"| formula={data.get('material_formula')} | method={data.get('growth_method')}"
        )
    elif event.event == "evidence_grade" and trace_enabled:
        print(
            "[evidence] "
            f"sufficient={data.get('is_sufficient')} | confidence={data.get('confidence')} "
            f"| {data.get('reason')}"
        )
    elif event.event == "citation" and trace_enabled:
        print(f"[citation] {data.get('record_id')} | doi={data.get('doi')}")
    elif event.event == "token":
        if not answer_chunks:
            print("\n答 > ", end="", flush=True)
        text = str(data.get("text", ""))
        answer_chunks.append(text)
        print(text, end="", flush=True)
    elif event.event == "error":
        print(f"\n[error] {json.dumps(data.get('errors', []), ensure_ascii=False)}", file=sys.stderr)


def ensure_trace_line_start(answer_chunks: list[str]) -> None:
    if answer_chunks and answer_chunks[-1] != "\n":
        print()
        answer_chunks.append("\n")


def print_final_metadata(final: dict[str, Any], printed_answer: bool) -> None:
    if not printed_answer and final.get("answer"):
        print(f"\n答 > {final['answer']}")
    elif printed_answer:
        print()

    citations = final.get("citations") or []
    if citations:
        print("\n[引用]")
        for item in citations:
            doi = item.get("doi") or "no-doi"
            print(f"- {item.get('record_id')} | {doi}")

    if final.get("errors"):
        print(f"\n[errors] {json.dumps(final['errors'], ensure_ascii=False)}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
