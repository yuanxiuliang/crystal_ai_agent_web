from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Protocol

from ..config import Settings, settings


class EmbeddingClient(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class LocalSentenceTransformerEmbeddingClient:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install service dependencies with `.venv/bin/python -m pip install -e .`."
            ) from exc

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Unable to determine embedding dimension for {self.model_name}.")
        return int(dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _resolve_device(self, requested: str) -> str | None:
        if requested and requested != "auto":
            return requested
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001 - fallback to library default device.
            return None
        return "cpu"


class LocalOnnxEmbeddingClient:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        *,
        batch_size: int,
        max_length: int,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise RuntimeError(
                f"ONNX embedding model not found: {model_file}. "
                "Run `rag-export-minilm-onnx` on a machine with enough memory first."
            )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime, numpy, and transformers are required for ONNX embeddings. "
                "Install dependencies with `.venv/bin/python -m pip install -e .`."
            ) from exc

        self.np = np
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        output_shape = self.session.get_outputs()[0].shape
        try:
            self._dimension = int(output_shape[-1])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unable to determine embedding dimension from {model_file}.") from exc

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        ort_inputs = {
            name: value
            for name, value in encoded.items()
            if name in self.input_names
        }
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"].astype("float32")
        vectors = self._mean_pool(token_embeddings, attention_mask)
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / self.np.clip(norms, 1e-12, None)
        return vectors.astype("float32").tolist()

    def _mean_pool(self, token_embeddings, attention_mask):
        mask = self.np.expand_dims(attention_mask, axis=-1)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = self.np.clip(mask.sum(axis=1), 1e-12, None)
        return summed / counts


@lru_cache(maxsize=1)
def get_default_embedding_client(config: Settings = settings) -> EmbeddingClient:
    if config.embedding_provider != "local":
        raise RuntimeError(f"Unsupported embedding provider: {config.embedding_provider}")
    if config.embedding_backend == "onnx":
        return LocalOnnxEmbeddingClient(
            config.embedding_onnx_model_path,
            config.embedding_tokenizer_path,
            batch_size=config.embedding_batch_size,
            max_length=config.embedding_max_length,
        )
    if config.embedding_backend != "torch":
        raise RuntimeError(f"Unsupported EMBEDDING_BACKEND: {config.embedding_backend}")
    return LocalSentenceTransformerEmbeddingClient(config.embedding_model, config.embedding_device)
