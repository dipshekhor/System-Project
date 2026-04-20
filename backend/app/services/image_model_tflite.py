"""
image_model_tflite.py
=====================
Loads and runs food image classification from a .tflite model.

This service is intentionally independent from verdict logic.
It only predicts {food_label, confidence} from an image.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services import food_lookup
from app.services.food_lookup import normalize_food_text

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - fallback path
    Interpreter = None  # type: ignore


_interpreter: Any = None
_input_details: dict[str, Any] | None = None
_output_details: dict[str, Any] | None = None
_class_names: list[str] | None = None
_loaded_model_path: str | None = None


def _candidate_model_paths() -> list[Path]:
    env_path = os.getenv("TEAMMATE_MODEL_PATH", "").strip()
    paths = []
    if env_path:
        paths.append(Path(env_path))

    # Phase 1 default: known compatible baseline model for runtime 2.14.0
    paths.append(Path("/app/model_assets/food_model_v214_baseline.tflite"))
    # Phase 2 candidate: quantized optimized model (swap via env when verified)
    paths.append(Path("/app/model_assets/food_model_v214_quantized.tflite"))
    # Accept common duplicate-download filename variant to avoid brittle deploys.
    paths.append(Path("/app/model_assets/food_model_v214_quantized (1).tflite"))
    # Backward-compatible legacy filename fallback
    paths.append(Path("/app/model_assets/food_model.tflite"))

    # Default local workspace fallback
    resolved = Path(__file__).resolve()
    root = resolved.parents[-1] if len(resolved.parents) > 0 else resolved.parent
    paths.append(root / "food_model_output_B3_v2-20260409T153143Z-3-001" / "food_model_output_B3_v2" / "food_model_v214_baseline.tflite")
    paths.append(root / "food_model_output_B3_v2-20260409T153143Z-3-001" / "food_model_output_B3_v2" / "food_model_v214_quantized.tflite")
    paths.append(root / "food_model_output_B3_v2-20260409T153143Z-3-001" / "food_model_output_B3_v2" / "food_model_v214_quantized (1).tflite")
    paths.append(root / "food_model_output_B3_v2-20260409T153143Z-3-001" / "food_model_output_B3_v2" / "food_model.tflite")
    paths.append(Path(__file__).resolve().parents[1] / "ml_models" / "food_model.tflite")
    return paths


def _candidate_class_names_paths() -> list[Path]:
    env_path = os.getenv("TEAMMATE_CLASS_NAMES_PATH", "").strip()
    paths = []
    if env_path:
        paths.append(Path(env_path))

    paths.append(Path("/app/model_assets/class_names.json"))

    resolved = Path(__file__).resolve()
    root = resolved.parents[-1] if len(resolved.parents) > 0 else resolved.parent
    paths.append(root / "food_model_output_B3_v2-20260409T153143Z-3-001" / "food_model_output_B3_v2" / "class_names.json")
    paths.append(Path(__file__).resolve().parents[1] / "ml_models" / "class_names.json")
    return paths


def _resolve_existing_path(candidates: list[Path], kind: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"{kind} file not found. Checked:\n{checked}")


def _softmax(vec: np.ndarray) -> np.ndarray:
    shifted = vec - np.max(vec)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _load_once() -> None:
    global _interpreter, _input_details, _output_details, _class_names, _loaded_model_path

    if _interpreter is not None and _class_names is not None:
        return

    if Interpreter is None:
        raise RuntimeError(
            "tflite-runtime interpreter is not available. "
            "Install tflite-runtime==2.14.0 in the backend environment."
        )

    class_names_path = _resolve_existing_path(_candidate_class_names_paths(), "Class names")

    with class_names_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, list) or not loaded:
        raise ValueError("class_names.json must be a non-empty JSON list")

    _class_names = [str(x) for x in loaded]

    try:
        model_path = _resolve_existing_path(_candidate_model_paths(), "TFLite model")
        _interpreter = Interpreter(model_path=str(model_path))
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()[0]
        _output_details = _interpreter.get_output_details()[0]
        _loaded_model_path = str(model_path)
    except Exception as e:
        raise RuntimeError(
            "Could not load TFLite image model. "
            "If error mentions unsupported opcode version (e.g., FULLY_CONNECTED v12), "
            "recreate model with TensorFlow 2.14 conversion pipeline. "
            f"Details: {str(e)}"
        )


def _normalize_predicted_label(raw_label: str) -> str:
    """
    Convert model class names into lookup-friendly food labels.
    """
    normalized = normalize_food_text(raw_label)

    aliases = {
        "sweetpotato": "sweet potato",
        "cup cakes": "cupcakes",
        "soy beans": "soy beans",
    }

    return aliases.get(normalized, normalized)


def _canonical_food_name(label: str) -> str:
    """
    Map model label to canonical food name from food_db using fuzzy lookup.
    Falls back to normalized label if DB lookup misses.
    """
    try:
        matched = normalize_food_text(label)
        hit = food_lookup.lookup(matched, threshold=55)
        if hit and isinstance(hit, dict) and hit.get("food_item"):
            return str(hit["food_item"])
    except Exception:
        pass
    return label


def preload() -> None:
    """Public startup hook: load interpreter + tensors + class names once."""
    _load_once()


def model_info() -> dict[str, Any]:
    """Expose lightweight runtime info for diagnostics."""
    _load_once()
    assert _input_details is not None
    return {
        "model_path": _loaded_model_path,
        "input_shape": list(_input_details["shape"]),
        "input_dtype": str(_input_details["dtype"]),
    }


def predict_from_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    """
    Predict food label from a raw image byte stream.

    Returns:
        {
          "food_label": "spaghetti bolognese",
          "confidence": 0.93,
          "raw_food_label": "spaghetti_bolognese"
        }
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")

    _load_once()
    assert _class_names is not None

    assert _interpreter is not None
    assert _input_details is not None
    assert _output_details is not None

    input_shape = _input_details["shape"]
    height = int(input_shape[1])
    width = int(input_shape[2])

    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((width, height))
    arr = np.array(image)

    input_dtype = _input_details["dtype"]
    if input_dtype == np.float32:
        # Model expects training-scale pixels [0, 255], not normalized [0, 1].
        tensor = arr.astype(np.float32)[np.newaxis, ...]
    else:
        tensor = arr.astype(input_dtype)[np.newaxis, ...]

    _interpreter.set_tensor(_input_details["index"], tensor)
    _interpreter.invoke()
    output = np.squeeze(_interpreter.get_tensor(_output_details["index"]))

    if output.ndim != 1:
        output = output.reshape(-1)

    # Handle either logits or probabilities
    if np.any(output < 0.0) or np.any(output > 1.0) or not np.isclose(np.sum(output), 1.0, atol=1e-3):
        probs = _softmax(output)
    else:
        probs = output

    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    if best_idx >= len(_class_names):
        raise RuntimeError(
            f"Predicted class index {best_idx} exceeds class_names length {len(_class_names)}"
        )

    raw_label = _class_names[best_idx]
    normalized_label = _normalize_predicted_label(raw_label)
    food_label = _canonical_food_name(normalized_label)

    return {
        "food_label": food_label,
        "confidence": confidence,
        "raw_food_label": raw_label,
        "normalized_food_label": normalized_label,
        "confidence_percentage": round(confidence * 100.0, 2),
    }
