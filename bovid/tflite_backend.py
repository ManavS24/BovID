"""Torch-free TFLite classifier runner. Exists because it imports no torch/ultralytics (the edge
path) and because Ultralytics' default XNNPACK delegate can't load the quantised graph.

Preprocessing exactly mirrors Ultralytics' classification transform; parity is enforced by
scripts/compare_classifiers.py so a mismatch can't be mistaken for quantisation loss.
"""

import os

import cv2
import numpy as np
import yaml
from PIL import Image


def load_interpreter(model_path, num_threads=None):
    """Build a TFLite interpreter with the default (XNNPACK) delegate disabled. Prefers
    ai_edge_litert, falls back to tf.lite. num_threads caps cores for honest edge benchmarking."""
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import OpResolverType

    kwargs = {"experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
    if num_threads is not None:
        kwargs["num_threads"] = num_threads

    interpreter = Interpreter(model_path=model_path, **kwargs)
    interpreter.allocate_tensors()
    return interpreter


# Export suffixes to strip to recover <base> (metadata.yaml lives in <base>_saved_model/). Not
# split("_")[0], which mangles a base name containing an underscore (yolov8s_cls_int8 -> yolov8s).
_EXPORT_SUFFIXES = ("_float32", "_float16", "_int8", "_full_integer_quant", "_integer_quant")


def _base_stem(model_path):
    """Recover the original weights stem from an exported filename by stripping a known export
    suffix (e.g. best_int8.tflite -> best, my_cls_model_float32.tflite -> my_cls_model)."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for suffix in _EXPORT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_metadata(model_path):
    """Locate the metadata.yaml Ultralytics wrote beside the export -- it carries the class names,
    which are not in the .tflite itself."""
    stem = _base_stem(model_path)
    directory = os.path.dirname(os.path.abspath(model_path))
    candidates = [
        os.path.join(directory, f"{stem}_saved_model", "metadata.yaml"),
        os.path.join(directory, "metadata.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no metadata.yaml found for {model_path} (looked in {candidates}); "
        "class names come from it, so inference cannot name its predictions without it"
    )


class TFLiteClassifier:
    """Runs an exported YOLOv8-cls TFLite model. Handles FP32 and INT8 (quantised I/O)."""

    def __init__(self, model_path, metadata_path=None, num_threads=None):
        self.model_path = model_path
        self.interpreter = load_interpreter(model_path, num_threads)
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        _, self.height, self.width, _ = self.input_detail["shape"]
        with open(metadata_path or _find_metadata(model_path)) as f:
            metadata = yaml.safe_load(f)
        self.names = metadata["names"]

    def preprocess(self, img_bgr):
        """BGR uint8 image -> the model's input tensor. Uses PIL to match Ultralytics' transform
        bit-for-bit (torchvision Resize+CenterCrop delegate to PIL; OpenCV resize diverges)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # torchvision Resize(int) scales the SHORTEST edge to that size, truncating the other.
        w, h = image.size
        if h < w:
            new_h, new_w = self.height, int(self.height * w / h)
        else:
            new_w, new_h = self.width, int(self.width * h / w)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # torchvision CenterCrop rounds rather than floors; off-by-one shifts predictions.
        left = int(round((new_w - self.width) / 2.0))
        top = int(round((new_h - self.height) / 2.0))
        image = image.crop((left, top, left + self.width, top + self.height))

        x = np.asarray(image, dtype=np.float32) / 255.0

        # An INT8 model takes quantised input: real_value = (q - zero_point) * scale, inverted.
        if self.input_detail["dtype"] == np.int8:
            scale_q, zero_point = self.input_detail["quantization"]
            x = np.round(x / scale_q + zero_point).clip(-128, 127).astype(np.int8)
        return x[None, ...]

    def predict(self, img_bgr):
        """Return the class-probability vector for one BGR image."""
        self.interpreter.set_tensor(self.input_detail["index"], self.preprocess(img_bgr))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_detail["index"])[0]

        if self.output_detail["dtype"] == np.int8:
            scale_q, zero_point = self.output_detail["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale_q
        return y.astype(np.float32)

    def top_k(self, img_bgr, k=3):
        """[(breed, confidence)] for the k most likely breeds, highest first."""
        probs = self.predict(img_bgr)
        order = probs.argsort()[::-1][:k]
        return [(self.names[int(i)], float(probs[i])) for i in order]
