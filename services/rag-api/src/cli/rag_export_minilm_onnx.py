from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..config import PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2-int8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-export-minilm-onnx",
        description="Export all-MiniLM-L6-v2 to an int8 ONNX model for low-resource CPU servers.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory for exported ONNX int8 model and tokenizer files.",
    )
    parser.add_argument(
        "--quantization",
        choices=["arm64", "avx2", "avx512", "avx512_vnni"],
        default="avx2",
        help="ONNX Runtime dynamic quantization preset.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the source model if it is not already cached locally.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint used when --allow-download is set.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    if args.allow_download:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    model = SentenceTransformer(
        args.model,
        backend="onnx",
        model_kwargs={"export": True},
        local_files_only=not args.allow_download,
    )
    export_dynamic_quantized_onnx_model(
        model,
        args.quantization,
        str(output_dir),
        push_to_hub=False,
    )
    model.tokenizer.save_pretrained(output_dir)
    print(f"output={output_dir}")
    print(f"quantization={args.quantization}")
    onnx_files = sorted((output_dir / "onnx").glob(f"*_{args.quantization}.onnx"))
    if onnx_files:
        print(f"onnx_model={onnx_files[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
