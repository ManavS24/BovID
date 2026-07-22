"""Export the trained classifier to deployment formats (ONNX / TFLite / SavedModel). Each format
is attempted independently so one unavailable backend doesn't cost the others.
"""

import logging
import os
import shutil

from ultralytics import YOLO

from . import config

logger = logging.getLogger(__name__)

# name -> ultralytics export kwargs. int8 calibrates on split="train" (the val split is too small
# and collapses the model); FP16 is exported but not the edge default (its input tensor can't run
# on CPU). See METHODOLOGY §2.
EXPORT_FORMATS = {
    "onnx": dict(format="onnx"),
    "tflite_fp32": dict(format="tflite", half=False),
    "tflite_fp16": dict(format="tflite", half=True),
    "tflite_int8": dict(format="tflite", int8=True, data=config.YOLO_DATA_DIR, split="train"),
    "saved_model": dict(format="saved_model"),
}


# Ultralytics expects "<stem>_int8.tflite" but current onnx2tf writes these names; the export
# succeeds, Ultralytics just returns a nonexistent path. Most-quantised first.
_INT8_ALIASES = ("_full_integer_quant.tflite", "_integer_quant.tflite")


def _resolve_int8(src):
    """Find the INT8 artifact onnx2tf actually wrote, given the name Ultralytics expected."""
    stem = os.path.basename(src)[: -len("_int8.tflite")]
    for alias in _INT8_ALIASES:
        candidate = os.path.join(os.path.dirname(src), stem + alias)
        if os.path.exists(candidate):
            logger.info("int8 artifact found under onnx2tf's name: %s", candidate)
            return candidate
    return None


def _copy_into(src, dest_dir):
    """Place an exported artifact in dest_dir, returning its final path (no-op when it is already
    there, the normal in-place export case)."""
    if not src:
        return None
    src = str(src)
    if not os.path.exists(src) and src.endswith("_int8.tflite"):
        src = _resolve_int8(src)
        if not src:
            return None
    if not os.path.exists(src):
        logger.warning("export reported success but no artifact exists at %s", src)
        return None

    dst = os.path.join(dest_dir, os.path.basename(src.rstrip("/")))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst  # already in place

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return dst


def export_model(weights=None, formats=None, imgsz=None, out_dir=None, model=None):
    """Export `weights` to each requested format, returning {name: path or None}. An unavailable
    backend yields None (logged) rather than aborting the others. `model` overrides weight loading
    (default YOLO(weights)) so tests can drive the orchestration without a real export."""
    weights = weights or config.DEFAULT_MODEL_PATH
    formats = formats or list(EXPORT_FORMATS)
    imgsz = imgsz or config.DEFAULT_IMGSZ
    out_dir = out_dir or config.BEST_MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)

    unknown = set(formats) - set(EXPORT_FORMATS)
    if unknown:
        raise ValueError(f"unknown export format(s): {sorted(unknown)}; "
                         f"choose from {sorted(EXPORT_FORMATS)}")

    if model is None:
        model = YOLO(weights)
    produced = {}
    for name in formats:
        try:
            path = model.export(imgsz=imgsz, **EXPORT_FORMATS[name])
            produced[name] = _copy_into(path, out_dir)
            logger.info("exported %s -> %s", name, produced[name])
        except Exception as e:
            # Broad by design: backends fail many ways, and one must not lose the others.
            logger.warning("%s export failed: %s", name, e)
            produced[name] = None
    return produced
