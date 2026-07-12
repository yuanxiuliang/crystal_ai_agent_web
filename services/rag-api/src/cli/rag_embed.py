from __future__ import annotations

import argparse
import math

from ..config import settings
from ..retrieval.embedding import get_default_embedding_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-embed",
        description="Smoke test the local embedding model used by RAG retrieval.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=[
            "ZnIn2S4 的 CVT 生长温度是多少？",
            "ZnIn2S4 single crystals grown by chemical vapor transport at source zone 900 C and crystal zone 850 C.",
            "Cs2AgBiBr6 crystals grown by flux method.",
        ],
        help="Texts to embed. Defaults to a small growth-method similarity smoke test.",
    )
    return parser


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def main() -> int:
    args = build_parser().parse_args()
    print(f"provider={settings.embedding_provider}")
    print(f"model={settings.embedding_model}")
    print(f"backend={settings.embedding_backend}")
    print(f"onnx_model_path={settings.embedding_onnx_model_path}")
    print(f"tokenizer_path={settings.embedding_tokenizer_path}")
    print(f"expected_dim={settings.embedding_dim}")
    print(f"batch_size={settings.embedding_batch_size}")
    print(f"max_length={settings.embedding_max_length}")
    print(f"device={settings.embedding_device}")

    client = get_default_embedding_client()
    vectors = client.embed_texts(args.texts)
    print(f"actual_dim={len(vectors[0]) if vectors else 0}")

    for index, text in enumerate(args.texts):
        print(f"[{index}] {text}")
    if len(vectors) >= 2:
        for index in range(1, len(vectors)):
            print(f"cosine(0,{index})={cosine(vectors[0], vectors[index]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
