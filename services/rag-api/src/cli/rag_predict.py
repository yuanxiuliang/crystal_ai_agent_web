from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ..prediction import (
    PredictionExecutionRequest,
    PredictionInputError,
    PredictionUnavailableError,
    PredictionValidationError,
    get_default_prediction_service,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Growth Route Transformer predictor."
    )
    parser.add_argument(
        "--formula", required=True, help="Target chemical formula, for example Mn3GaN."
    )
    parser.add_argument(
        "--user-id", default="cli-user", help="Development owner for prediction-run storage."
    )
    parser.add_argument(
        "--session-id", default=None, help="Optional development session reference."
    )
    parser.add_argument(
        "--message-id", default=None, help="Optional source chat-message reference."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete structured result as JSON."
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    service = get_default_prediction_service()
    try:
        result = await service.predict(
            PredictionExecutionRequest(
                user_id=args.user_id,
                session_id=args.session_id,
                message_id=args.message_id,
                formula=args.formula,
                source="explicit_prediction",
            )
        )
    except (PredictionInputError, PredictionUnavailableError, PredictionValidationError) as exc:
        print(f"[prediction:error] {exc}", file=sys.stderr)
        return 1

    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(
        f"prediction_run_id={result.prediction_run_id}\n"
        f"formula={result.formula_std}\n"
        f"model={result.model.model_id}@{result.model.model_version}\n"
        f"runtime_ms={result.runtime_ms}"
    )
    for route in result.routes:
        reactants = ", ".join(item["name"] for item in route.raw_reactants) or "none"
        additives = ", ".join(item["name"] for item in route.additives) or "none"
        print(
            f"route={route.rank} method={route.method} relative_rank_weight={route.relative_rank_weight:.4f}\n"
            f"  raw_reactants={reactants}\n"
            f"  additives={additives}\n"
            f"  growth={json.dumps(route.growth, ensure_ascii=False)}"
        )
    for warning in result.warnings:
        print(f"warning={warning}")
    return 0


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
