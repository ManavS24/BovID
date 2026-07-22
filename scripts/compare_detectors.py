"""Compare crop detectors on breed accuracy, not detection rate (a missed detection just takes
the fallback crop, so detection rate is misleading). Reports per detector: top-1/top-3 of the
real classifier on its crops, agreement + IoU vs the baseline, detection rate, latency, size.

    python scripts/compare_detectors.py --detectors models/pretrained/yolov8x.pt \
        models/pretrained/yolov8n.pt --limit 200
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from bovid import config, crop
from bovid.dataset import labelled_images
from bovid.logging_conf import setup_logging
from bovid.predict import load_model
from bovid.utils import bgr_to_pil, load_image_bgr

logger = logging.getLogger(__name__)


def _iou(a, b):
    """IoU of two (x1,y1,x2,y2) boxes; 0.0 if either is missing."""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _load_detector(path):
    """Load a detector; task='detect' explicit (Ultralytics guesses wrong for exported formats)."""
    from ultralytics import YOLO
    return YOLO(path, task="detect")


def _detect_box(detector, img_bgr, path):
    """Best animal box, or None. device= only for .pt (TFLite/ONNX have no 'mps' provider)."""
    h, w = img_bgr.shape[:2]
    kwargs = dict(verbose=False, conf=config.DETECT_CONF)
    if path.endswith(".pt"):
        kwargs["device"] = config.DEVICE
    result = detector(img_bgr, **kwargs)[0]
    return crop._box_from_result(result, w, h)


def evaluate_detector(path, images, classifier):
    """Run one detector over `images` and score the classifier on the crops it produces."""
    detector = _load_detector(path)
    records, detect_ms = [], []

    for image_path, breed in images:
        img_bgr = load_image_bgr(image_path)

        t0 = time.perf_counter()
        box = _detect_box(detector, img_bgr, path)
        detect_ms.append((time.perf_counter() - t0) * 1000)

        crop_bgr, method = crop.crop_from_box(img_bgr, box)
        r = classifier.predict(source=bgr_to_pil(crop_bgr), imgsz=config.DEFAULT_IMGSZ,
                               verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        order = probs.argsort()[::-1][:3]
        records.append(dict(
            image_path=image_path,
            true_breed=breed,
            box=box,
            method=method,
            top1=r.names[int(order[0])],
            top1_conf=float(probs[order[0]]),
            top3=[r.names[int(i)] for i in order],
        ))

    size_mb = (os.path.getsize(path) if os.path.isfile(path) else
               sum(os.path.getsize(os.path.join(d, f))
                   for d, _, fs in os.walk(path) for f in fs)) / 1e6
    return records, dict(
        detector=path,
        size_mb=round(size_mb, 1),
        detect_rate=round(np.mean([r["method"] == "detector" for r in records]), 3),
        top1=round(np.mean([r["top1"] == r["true_breed"] for r in records]), 3),
        top3=round(np.mean([r["true_breed"] in r["top3"] for r in records]), 3),
        detect_ms=round(float(np.median(detect_ms)), 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", nargs="+", required=True,
                        help="Detector paths; the FIRST is the baseline others are compared to.")
    parser.add_argument("--split", default="test",
                        help="Roboflow split to sample. Keep this HELD OUT -- see the check below.")
    parser.add_argument("--limit", type=int, default=200, help="Images to sample (stratified).")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH, help="Classifier weights.")
    parser.add_argument("--out", default=None, help="Write the summary table as JSON here.")
    args = parser.parse_args()

    setup_logging()
    images = labelled_images(args.split, args.limit)
    logger.info("comparing %d detector(s) over %d images from split %r",
                len(args.detectors), len(images), args.split)

    classifier = load_model(args.model)
    baseline_records, summaries = None, []

    for path in args.detectors:
        logger.info("evaluating %s", path)
        records, summary = evaluate_detector(path, images, classifier)

        if baseline_records is None:
            baseline_records = records
            summary["agree_with_baseline"] = 1.0
            summary["median_iou_vs_baseline"] = 1.0
        else:
            summary["agree_with_baseline"] = round(
                np.mean([a["top1"] == b["top1"] for a, b in zip(records, baseline_records, strict=True)]), 3)
            # Only where BOTH detectors fired (a shared fallback would score IoU 0 and mislead).
            ious = [_iou(a["box"], b["box"]) for a, b in zip(records, baseline_records, strict=True)
                    if a["box"] is not None and b["box"] is not None]
            summary["median_iou_vs_baseline"] = round(float(np.median(ious)), 3) if ious else None
            summary["both_detected"] = len(ious)
        summaries.append(summary)

    table = pd.DataFrame(summaries)
    print("\n" + table.to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summaries, f, indent=2)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
