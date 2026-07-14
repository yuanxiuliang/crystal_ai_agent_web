from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import PredictionModelInfo, PredictionUnavailableError


@dataclass(frozen=True)
class PredictionModelBundle:
    root: Path
    manifest: dict[str, Any]
    checkpoint_path: Path

    @classmethod
    def load(cls, model_dir: str | Path) -> "PredictionModelBundle":
        root = Path(model_dir).expanduser().resolve()
        manifest_path = root / "MANIFEST.json"
        if not manifest_path.is_file():
            raise PredictionUnavailableError(f"Prediction manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PredictionUnavailableError(
                f"Prediction manifest is invalid JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise PredictionUnavailableError("Prediction manifest must be a JSON object.")

        for key in ("model_id", "model_version", "checkpoint", "checkpoint_sha256", "task"):
            if not manifest.get(key):
                raise PredictionUnavailableError(f"Prediction manifest is missing {key!r}.")
        task = manifest["task"]
        if not isinstance(task, dict) or not isinstance(task.get("supported_methods"), list):
            raise PredictionUnavailableError("Prediction manifest task contract is invalid.")

        checkpoint_path = (root / str(manifest["checkpoint"])).resolve()
        try:
            checkpoint_path.relative_to(root)
        except ValueError as exc:
            raise PredictionUnavailableError(
                "Prediction checkpoint must remain inside the model bundle."
            ) from exc
        if not checkpoint_path.is_file():
            raise PredictionUnavailableError(f"Prediction checkpoint is missing: {checkpoint_path}")

        for relative in (
            "src/predict.py",
            "src/library.py",
            "src/network.py",
            "features/input_vocab.json",
            "features/output_vocab.json",
            "lib/config/token_meta.json",
            "lib/config/bucket_config.json",
            "lib/rawLib/reactant_element_map.json",
        ):
            if not (root / relative).is_file():
                raise PredictionUnavailableError(
                    f"Prediction runtime asset is missing: {root / relative}"
                )
        return cls(root=root, manifest=manifest, checkpoint_path=checkpoint_path)

    @property
    def model_info(self) -> PredictionModelInfo:
        task = self.manifest["task"]
        return PredictionModelInfo(
            model_id=str(self.manifest["model_id"]),
            model_version=str(self.manifest["model_version"]),
            artifact_digest=str(self.manifest["checkpoint_sha256"]),
            supported_methods=[str(item) for item in task["supported_methods"]],
            parameter_count=int(self.manifest.get("parameter_count", 0)),
        )

    def verify_checkpoint_digest(self) -> None:
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        expected = str(self.manifest["checkpoint_sha256"]).lower()
        actual = digest.hexdigest().lower()
        if actual != expected:
            raise PredictionUnavailableError(
                "Prediction checkpoint digest mismatch; refusing to load the model artifact. "
                f"expected={expected} actual={actual}"
            )
