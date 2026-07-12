from __future__ import annotations

import argparse
import asyncio

from ..memory.worker import MemoryWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-memory-worker",
        description="Process bounded long-term-memory background jobs.",
    )
    parser.add_argument("--once", action="store_true", help="Process one batch then exit.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum jobs in one batch.")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    worker = MemoryWorker.from_settings()
    try:
        if args.once:
            processed = await worker.run_once(args.limit)
            print(f"processed={processed}")
            return 0
        await worker.run_forever()
        return 0
    finally:
        await worker.close()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
