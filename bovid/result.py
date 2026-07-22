"""The result contract shared by the server and edge paths.

Separate from predict.py (which imports torch at module scope) so the torch-free edge path can
return the same objects.
"""

from dataclasses import dataclass
from typing import NamedTuple

from . import config


class Prediction(NamedTuple):
    breed: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    image_path: str
    predictions: list          # list[Prediction], highest confidence first
    crop_method: str           # "detector" or "fallback"
    crop_bgr: object = None    # the exact crop fed to the classifier

    @property
    def top(self):
        return self.predictions[0]

    @property
    def is_confident(self):
        """Top-1 clears the measured CONFIDENCE_THRESHOLD; below it, surface as uncertain."""
        return self.top.confidence >= config.CONFIDENCE_THRESHOLD
