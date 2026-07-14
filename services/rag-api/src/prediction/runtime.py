from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..config import Settings, settings
from .bundle import PredictionModelBundle
from .contracts import FormulaValidation, PredictionInputError, PredictionUnavailableError


_VENDOR_IMPORT_LOCK = threading.Lock()
_VENDOR_PREFIX = "_agentweb_growth_route_transformer_v2"


@dataclass(frozen=True)
class _VendorBindings:
    module: ModuleType


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PredictionUnavailableError(f"Unable to load vendored prediction module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_vendor_predictor(source_dir: Path) -> _VendorBindings:
    """Load the preserved predictor source without permanently exposing generic module names."""
    private_predict_name = f"{_VENDOR_PREFIX}.predict"
    existing = sys.modules.get(private_predict_name)
    if existing is not None:
        return _VendorBindings(module=existing)

    with _VENDOR_IMPORT_LOCK:
        existing = sys.modules.get(private_predict_name)
        if existing is not None:
            return _VendorBindings(module=existing)

        library_path = source_dir / "library.py"
        network_path = source_dir / "network.py"
        predict_path = source_dir / "predict.py"
        if not all(path.is_file() for path in (library_path, network_path, predict_path)):
            raise PredictionUnavailableError("Vendored prediction source is incomplete.")

        private_library_name = f"{_VENDOR_PREFIX}.library"
        private_network_name = f"{_VENDOR_PREFIX}.network"
        library = _load_module(private_library_name, library_path)
        network = _load_module(private_network_name, network_path)

        aliases = {"library": library, "network": network}
        previous = {name: sys.modules.get(name) for name in aliases}
        for name, current in previous.items():
            if current is not None:
                current_path = Path(str(getattr(current, "__file__", ""))).resolve()
                expected_path = (source_dir / f"{name}.py").resolve()
                if current_path != expected_path:
                    raise PredictionUnavailableError(
                        f"Prediction runtime import namespace conflict for module {name!r}."
                    )
        try:
            sys.modules.update(aliases)
            predictor = _load_module(private_predict_name, predict_path)
        except ImportError as exc:
            raise PredictionUnavailableError(
                "Prediction runtime dependencies are unavailable. Install rag-api with the prediction extra."
            ) from exc
        finally:
            for name, prior in previous.items():
                if prior is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior
        return _VendorBindings(module=predictor)


class PredictionRuntime:
    """Lazy, single-process CPU runtime for the versioned route-prediction bundle."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        bundle: PredictionModelBundle | None = None,
    ) -> None:
        self.config = config
        self.bundle = bundle or PredictionModelBundle.load(config.prediction_model_dir)
        self._context: Any | None = None
        self._context_lock = threading.Lock()
        self._vendor: _VendorBindings | None = None

    @property
    def is_loaded(self) -> bool:
        return self._context is not None

    def validate_formula(self, formula: str) -> FormulaValidation:
        value = formula.strip()
        if not value:
            raise PredictionInputError("Formula is required.")
        if len(value) > 160:
            raise PredictionInputError("Formula exceeds the 160-character limit.")
        vendor = self._get_vendor().module
        try:
            formula_std = str(vendor.normalize_formula_candidate(value))
            if not formula_std or vendor.parse_composition(formula_std) is None:
                raise ValueError("formula could not be parsed")
            tokens = [str(item) for item in vendor.tokenize_formula(formula_std)]
            elements = [str(item) for item in vendor.ordered_elements_from_formula(formula_std)]
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"Formula is not parseable: {value}") from exc
        if not tokens or not elements:
            raise PredictionInputError(f"Formula is not usable by the deployed model: {value}")
        context = self._context
        input_vocab = getattr(context, "input_vocab", None) if context is not None else None
        unknown = [
            token for token in tokens if input_vocab is not None and token not in input_vocab
        ]
        return FormulaValidation(
            formula=value,
            formula_std=formula_std,
            formula_tokens=tokens,
            target_elements=elements,
            unknown_formula_tokens=unknown,
        )

    def generate(self, formula: str) -> dict[str, Any]:
        context = self._get_or_load_context()
        vendor = self._get_vendor().module
        try:
            return dict(
                vendor.generate_routes_with_context(
                    formula=formula,
                    beam_size=max(1, self.config.prediction_beam_size),
                    num_return_sequences=max(1, min(3, self.config.prediction_return_sequences)),
                    context=context,
                )
            )
        except ValueError as exc:
            raise PredictionInputError(str(exc)) from exc

    def _get_vendor(self) -> _VendorBindings:
        if self._vendor is None:
            self._vendor = _load_vendor_predictor(self.bundle.root / "src")
        return self._vendor

    def _get_or_load_context(self) -> Any:
        if self._context is not None:
            return self._context
        with self._context_lock:
            if self._context is not None:
                return self._context
            if self.config.prediction_device.strip().lower() != "cpu":
                raise PredictionUnavailableError(
                    "The deployed growth-route model currently supports PREDICTION_DEVICE=cpu only."
                )
            self.bundle.verify_checkpoint_digest()
            try:
                import torch
            except ImportError as exc:
                raise PredictionUnavailableError(
                    "Prediction runtime requires PyTorch. Install rag-api with the prediction extra."
                ) from exc
            torch.set_num_threads(max(1, min(8, self.config.prediction_torch_threads)))
            vendor = self._get_vendor().module
            try:
                self._context = vendor.load_prediction_context(
                    run_name=self.bundle.model_info.model_version,
                    checkpoint=str(self.bundle.checkpoint_path),
                    device="cpu",
                )
            except Exception as exc:  # noqa: BLE001 - convert vendored runtime failures at the boundary.
                raise PredictionUnavailableError(
                    f"Unable to load prediction checkpoint: {type(exc).__name__}: {exc}"
                ) from exc
            return self._context
