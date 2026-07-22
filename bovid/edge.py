"""Torch-free edge inference: same pipeline and PredictionResult contract as bovid.predict, but
TFLite detector + classifier with no torch/ultralytics (which don't install on target devices).

Shares the crop policy with the server path so both crop identically. Edge artifacts are FP32
(INT8 was measured and rejected -- METHODOLOGY §2).

    result = EdgePipeline().predict("cow.jpg")
"""

import logging
import os

from . import config
from .crop import crop_from_box
from .result import Prediction, PredictionResult
from .tflite_backend import TFLiteClassifier
from .tflite_detector import TFLiteDetector
from .utils import load_image_bgr

logger = logging.getLogger(__name__)

DEFAULT_EDGE_CLASSIFIER = os.path.join(config.BEST_MODEL_DIR, "best_float32.tflite")


class EdgePipeline:
    """Detector + classifier, both TFLite. Load once and reuse: interpreter construction is the
    expensive part."""

    def __init__(self, detector_path=None, classifier_path=None, num_threads=None):
        self.detector = TFLiteDetector(detector_path, num_threads=num_threads)
        self.classifier = TFLiteClassifier(classifier_path or DEFAULT_EDGE_CLASSIFIER,
                                           num_threads=num_threads)
        logger.info("edge pipeline ready: detector=%s classifier=%s",
                    os.path.basename(self.detector.model_path),
                    os.path.basename(self.classifier.model_path))

    def predict(self, image_path, topk=None):
        """Classify one image, returning the same PredictionResult as the server path."""
        topk = topk or config.DEFAULT_TOPK
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = load_image_bgr(image_path)
        crop_bgr, crop_method = crop_from_box(img_bgr, self.detector.best_animal_box(img_bgr))
        predictions = [Prediction(breed, confidence)
                       for breed, confidence in self.classifier.top_k(crop_bgr, k=topk)]

        return PredictionResult(
            image_path=image_path,
            predictions=predictions,
            crop_method=crop_method,
            crop_bgr=crop_bgr,
        )

    def predict_batch(self, image_paths, topk=None):
        return [self.predict(path, topk=topk) for path in image_paths]
