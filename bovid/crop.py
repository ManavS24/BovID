"""Animal crop shared by training-prep and inference: detector crop, else a deterministic
bottom/edge trim. See METHODOLOGY §2-3.
"""


from . import config

# cow is the target; the rest catch bovine poses labelled as a similar large animal.
ANIMAL_CLASSES = {"cow", "horse", "sheep", "elephant", "dog", "cat", "bird"}

_detector = None


def get_detector():
    # task="detect" explicit: Ultralytics only guesses it for exported (.tflite) formats.
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(config.DETECTOR_MODEL, task="detect")
    return _detector


def select_animal_box(candidates, w, h):
    """Largest animal box passing the class + coverage gates, or None. Plain tuples so the torch
    and torch-free detectors share one copy of this policy."""
    best_box, best_area = None, 0.0
    for name, x1, y1, x2, y2 in candidates:
        if name not in ANIMAL_CLASSES:
            continue
        area = (x2 - x1) * (y2 - y1)
        coverage = area / (w * h)
        if not (config.DETECT_MIN_COV <= coverage <= config.DETECT_MAX_COV):
            continue
        if area > best_area:
            best_area, best_box = area, (int(x1), int(y1), int(x2), int(y2))
    return best_box


def _box_from_result(result, w, h):
    candidates = [
        (result.names[int(box.cls)], *(float(v) for v in box.xyxy[0]))
        for box in result.boxes
    ]
    return select_animal_box(candidates, w, h)


def _best_animal_box(img_bgr, detector):
    # device= only for .pt checkpoints -- TFLite/ONNX runtimes have no "mps" provider.
    h, w = img_bgr.shape[:2]
    kwargs = dict(verbose=False, conf=config.DETECT_CONF)
    if str(detector.ckpt_path).endswith(".pt"):
        kwargs["device"] = config.DEVICE
    result = detector(img_bgr, **kwargs)[0]
    return _box_from_result(result, w, h)


def _pad_box(x1, y1, x2, y2, w, h):
    # Bottom padded only by DETECT_BOTTOM_PAD (0): extending below re-adds the overlay margin.
    pw = int((x2 - x1) * config.DETECT_PAD)
    ph = int((y2 - y1) * config.DETECT_PAD)
    bottom_ph = int((y2 - y1) * config.DETECT_BOTTOM_PAD)
    return max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + bottom_ph)


def _deterministic_crop(img_bgr):
    """Trim bottom/edge margins (where most overlays live), then centre-square crop."""
    h, w = img_bgr.shape[:2]
    top = int(h * config.FALLBACK_EDGE_TRIM)
    bottom = int(h * (1 - config.FALLBACK_BOTTOM_TRIM))
    left = int(w * config.FALLBACK_EDGE_TRIM)
    right = int(w * (1 - config.FALLBACK_EDGE_TRIM))
    region = img_bgr[top:bottom, left:right]

    rh, rw = region.shape[:2]
    side = min(rh, rw)
    y0, x0 = (rh - side) // 2, (rw - side) // 2
    return region[y0:y0 + side, x0:x0 + side]


def crop_from_box(img_bgr, box):
    """Apply a detected box (min 32px), else fall back. Shared by server and edge paths."""
    if box is not None:
        x1, y1, x2, y2 = _pad_box(*box, img_bgr.shape[1], img_bgr.shape[0])
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size and crop.shape[0] >= 32 and crop.shape[1] >= 32:
            return crop, "detector"
    return _deterministic_crop(img_bgr), "fallback"


def crop_animal(img_bgr, detector=None):
    """Return (crop_bgr, method) where method is 'detector' or 'fallback'."""
    detector = detector or get_detector()
    return crop_from_box(img_bgr, _best_animal_box(img_bgr, detector))


# Per-image, not batched: batching produced different crops near the gates (train/serve skew).
