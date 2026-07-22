"""Central configuration: paths and training/inference hyperparameters.

Every constant is overridable via a `BOVID_<NAME>` environment variable and validated at import
(logical invariants only; resource existence is left to each consumer).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_root(env_name, dir_name):
    """Locate models/ or datasets/ (not shipped in the wheel): BOVID_<env_name> override, else a
    repo checkout, else the cwd. Without this a non-editable install resolves them into
    site-packages and fails."""
    override = os.environ.get(f"BOVID_{env_name}")
    if override:
        return os.path.abspath(override)

    in_repo = os.path.join(PROJECT_ROOT, dir_name)
    if os.path.isdir(in_repo):
        return in_repo
    return os.path.abspath(dir_name)


DATA_ROOT = _resolve_root("DATA_ROOT", "datasets")
MODELS_ROOT = _resolve_root("MODELS_ROOT", "models")


# --- typed environment overrides -------------------------------------------------------------
def _env_str(name, default):
    return os.environ.get(f"BOVID_{name}", default)


def _env_float(name, default):
    raw = os.environ.get(f"BOVID_{name}")
    return default if raw is None else float(raw)


def _env_int(name, default):
    raw = os.environ.get(f"BOVID_{name}")
    return default if raw is None else int(raw)


def _env_bool(name, default):
    raw = os.environ.get(f"BOVID_{name}")
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_device_cache = None


def __getattr__(name):
    """Resolve config.DEVICE lazily: detecting it imports torch, and doing that at import time
    would pull torch into the torch-free edge path. Set BOVID_DEVICE to skip the probe."""
    if name != "DEVICE":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    global _device_cache
    if _device_cache is None:
        _device_cache = os.environ.get("BOVID_DEVICE") or _default_device()
        _check_device(_device_cache)
    return _device_cache

# ---- Dataset ----
RAW_DATA_DIR = _env_str("RAW_DATA_DIR", os.path.join(DATA_ROOT, "indianbovine1"))
CROPS_ROOT = _env_str("CROPS_ROOT", os.path.join(DATA_ROOT, "bovine_crops"))
YOLO_DATA_DIR = _env_str("YOLO_DATA_DIR", os.path.join(DATA_ROOT, "bovine_cls"))

# ---- Model ----
# Absolute paths so Ultralytics resolves weights regardless of cwd (a bare name re-downloads).
PRETRAINED_DIR = os.path.join(MODELS_ROOT, "pretrained")
CLASSIFIER_MODEL = _env_str("CLASSIFIER_MODEL", os.path.join(PRETRAINED_DIR, "yolov8s-cls.pt"))
RUNS_DIR = _env_str("RUNS_DIR", os.path.join(PROJECT_ROOT, "runs"))
BEST_MODEL_DIR = _env_str("BEST_MODEL_DIR", os.path.join(MODELS_ROOT, "best"))
DEFAULT_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best.pt")

# ---- Training ----
DO_OVERSAMPLE_TRAIN = _env_bool("DO_OVERSAMPLE_TRAIN", True)
DO_FINETUNE_320 = _env_bool("DO_FINETUNE_320", True)
DO_WATERMARK_AUG = _env_bool("DO_WATERMARK_AUG", True)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)

STAGE1 = dict(imgsz=224, epochs=40, batch=32, lr0=0.001, patience=25)
STAGE2 = dict(imgsz=320, epochs=20, batch=32, lr0=5e-4, patience=15)

# ---- Cropping ----
# Server uses yolov8x; edge uses the tiny FP32 TFLite detector (FP16 can't run on CPU). See
# METHODOLOGY §2.
DETECTOR_MODEL = _env_str("DETECTOR_MODEL", os.path.join(PRETRAINED_DIR, "yolov8x.pt"))
EDGE_DETECTOR_MODEL = _env_str("EDGE_DETECTOR_MODEL",
                               os.path.join(PRETRAINED_DIR, "yolov8n_float32.tflite"))
DETECT_CONF = _env_float("DETECT_CONF", 0.25)
DETECT_MIN_COV = _env_float("DETECT_MIN_COV", 0.12)        # reject tiny spurious boxes
DETECT_MAX_COV = _env_float("DETECT_MAX_COV", 0.98)        # a near-full-frame box adds nothing
DETECT_PAD = _env_float("DETECT_PAD", 0.05)
DETECT_BOTTOM_PAD = _env_float("DETECT_BOTTOM_PAD", 0.0)   # not below the animal -- overlays live there
FALLBACK_BOTTOM_TRIM = _env_float("FALLBACK_BOTTOM_TRIM", 0.15)
FALLBACK_EDGE_TRIM = _env_float("FALLBACK_EDGE_TRIM", 0.05)

# ---- Inference ----
DEFAULT_IMGSZ = _env_int("DEFAULT_IMGSZ", 224)   # must match the classifier's training size
DEFAULT_TOPK = _env_int("DEFAULT_TOPK", 3)

# Top-1 below this is flagged uncertain; 0.9 measured (re-check with evaluate.py after a retrain).
CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.9)


def _check(cond, msg):
    if not cond:
        raise ValueError(f"invalid bovid config: {msg}")


def _check_device(device):
    _check(device == "cpu" or device == "mps" or device.startswith("cuda"),
           f"DEVICE must be cpu, mps or cuda[:N], got {device!r}")


def _validate():
    check = _check

    check(0.0 <= DETECT_CONF <= 1.0, f"DETECT_CONF must be in [0,1], got {DETECT_CONF}")
    check(0.0 <= DETECT_MIN_COV < DETECT_MAX_COV <= 1.0,
          f"need 0 <= DETECT_MIN_COV < DETECT_MAX_COV <= 1, "
          f"got min={DETECT_MIN_COV} max={DETECT_MAX_COV}")
    check(DETECT_PAD >= 0 and DETECT_BOTTOM_PAD >= 0,
          f"padding must be >= 0, got pad={DETECT_PAD} bottom_pad={DETECT_BOTTOM_PAD}")
    check(0.0 <= FALLBACK_BOTTOM_TRIM < 1.0,
          f"FALLBACK_BOTTOM_TRIM must be in [0,1), got {FALLBACK_BOTTOM_TRIM}")
    check(0.0 <= FALLBACK_EDGE_TRIM < 0.5,
          f"FALLBACK_EDGE_TRIM must be in [0,0.5), got {FALLBACK_EDGE_TRIM}")
    # a fallback crop must leave a non-empty region
    check(FALLBACK_BOTTOM_TRIM + FALLBACK_EDGE_TRIM < 1.0,
          "FALLBACK_BOTTOM_TRIM + FALLBACK_EDGE_TRIM must be < 1 or the fallback crop is empty")
    check(0.0 <= CONFIDENCE_THRESHOLD <= 1.0,
          f"CONFIDENCE_THRESHOLD must be in [0,1], got {CONFIDENCE_THRESHOLD}")
    check(DEFAULT_TOPK >= 1, f"DEFAULT_TOPK must be >= 1, got {DEFAULT_TOPK}")
    check(DEFAULT_IMGSZ > 0, f"DEFAULT_IMGSZ must be > 0, got {DEFAULT_IMGSZ}")


_validate()
