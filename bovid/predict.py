"""Server inference API: crop (shared with training) then classify."""

import os

import numpy as np
from ultralytics import YOLO

from . import config
from .crop import crop_animal
from .result import Prediction, PredictionResult
from .utils import bgr_to_pil, load_image_bgr

__all__ = ["Prediction", "PredictionResult", "load_model", "predict", "predict_batch"]

DEFAULT_MODEL_PATH = config.DEFAULT_MODEL_PATH
DEFAULT_IMGSZ = config.DEFAULT_IMGSZ
DEFAULT_TOPK = config.DEFAULT_TOPK


def load_model(model_path=DEFAULT_MODEL_PATH):
    # task="classify" explicit: Ultralytics guesses `detect` for exported formats and fails.
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    return YOLO(model_path, task="classify")


def predict(image_path, model=None, imgsz=DEFAULT_IMGSZ, topk=DEFAULT_TOPK):
    """Classify one image. Returns a PredictionResult (top-k breeds, crop method, confidence)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model is None:
        model = load_model()

    crop_bgr, crop_method = crop_animal(load_image_bgr(image_path))
    # No device= : ONNX/TFLite backends have no "mps" provider and raise on it.
    r = model.predict(source=bgr_to_pil(crop_bgr), imgsz=imgsz, verbose=False)[0]

    probs = r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu") else np.array(r.probs.data)
    order = probs.argsort()[::-1][:topk]
    return PredictionResult(
        image_path=image_path,
        predictions=[Prediction(r.names[int(i)], float(probs[i])) for i in order],
        crop_method=crop_method,
        crop_bgr=crop_bgr,
    )


def predict_batch(image_paths, model=None, imgsz=DEFAULT_IMGSZ, topk=DEFAULT_TOPK):
    # Sequential by design: batching shifts crops near the confidence gate (train/serve skew).
    model = model or load_model()
    return [predict(p, model=model, imgsz=imgsz, topk=topk) for p in image_paths]
