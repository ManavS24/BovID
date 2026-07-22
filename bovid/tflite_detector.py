"""Torch-free YOLOv8 detector on TFLite for the edge crop stage: letterbox, (1,84,8400) head
decode and NMS done in numpy (what Ultralytics would do in torch). Box selection is shared with
the server path via crop.select_animal_box.
"""

import os

import cv2
import numpy as np
import yaml

from . import config
from .crop import select_animal_box
from .tflite_backend import load_interpreter

NMS_IOU = 0.7   # Ultralytics' default, kept explicit so the edge box set matches the server's


def _load_names(model_path):
    """COCO class names, generated from the .pt checkpoint at export time (not hardcoded -- a
    wrong index would silently mispoint "cow")."""
    names_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "yolov8n_names.yaml")
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f"{names_path} not found; regenerate it from the detector checkpoint "
            "(it maps class indices to names and cannot be inferred from the .tflite)"
        )
    with open(names_path) as f:
        return yaml.safe_load(f)["names"]


def letterbox(img_bgr, size):
    """Aspect-preserving resize + pad to a square `size`, as Ultralytics does. Returns
    (padded, scale, pad_x, pad_y) so detections can be mapped back to original coordinates."""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)

    # INTER_AREA when shrinking, matching the antialiasing behaviour the model was trained with.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)   # 114 is Ultralytics' pad value
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold=NMS_IOU):
    """Greedy non-maximum suppression. Returns the indices to keep, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


class TFLiteDetector:
    """Runs an exported YOLOv8 detector on TFLite and returns boxes in original-image pixels."""

    def __init__(self, model_path=None, num_threads=None):
        self.model_path = model_path or config.EDGE_DETECTOR_MODEL
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"edge detector not found: {self.model_path}\n"
                "Weights are not shipped inside the package. Either run from a checkout that "
                "has models/, or point bovid at them:\n"
                "    BOVID_MODELS_ROOT=/path/to/models bovid predict --edge photo.jpg"
            )
        self.interpreter = load_interpreter(self.model_path, num_threads)
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        _, self.size, _, _ = self.input_detail["shape"]
        self.names = _load_names(self.model_path)

    def detect(self, img_bgr, conf=None):
        """Return [(name, x1, y1, x2, y2, score)] in the ORIGINAL image's coordinates."""
        conf = config.DETECT_CONF if conf is None else conf
        h, w = img_bgr.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(img_bgr, self.size)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input_detail["index"], rgb[None, ...])
        self.interpreter.invoke()
        # (1, 84, 8400) -> (8400, 84): 4 box coords (cx, cy, w, h) then 80 class scores.
        raw = self.interpreter.get_tensor(self.output_detail["index"])[0].T

        boxes_cxcywh, class_scores = raw[:, :4], raw[:, 4:]
        best_class = class_scores.argmax(axis=1)
        best_score = class_scores.max(axis=1)

        keep_mask = best_score >= conf
        if not keep_mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[keep_mask]
        best_class, best_score = best_class[keep_mask], best_score[keep_mask]

        # The TFLite export emits box coordinates normalised to 0-1; scale to input pixels.
        boxes_cxcywh = boxes_cxcywh * self.size

        cx, cy, bw, bh = (boxes_cxcywh[:, i] for i in range(4))
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        # Undo the letterbox: remove padding, then the resize scale.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        return [
            (self.names[int(best_class[i])], *boxes[i].tolist(), float(best_score[i]))
            for i in nms(boxes, best_score)
        ]

    def best_animal_box(self, img_bgr):
        """The box the crop stage should use, or None (gating shared via crop.select_animal_box)."""
        h, w = img_bgr.shape[:2]
        candidates = [(name, x1, y1, x2, y2)
                      for name, x1, y1, x2, y2, _ in self.detect(img_bgr)]
        return select_animal_box(candidates, w, h)
