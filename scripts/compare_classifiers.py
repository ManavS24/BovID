"""Compare classifier artifacts (.pt / .onnx / .tflite) on accuracy, size and latency, scoring
every artifact on identical crops (cropped once, reused) so only the model format varies.

    python scripts/compare_classifiers.py --models models/best/best.pt \
        models/best/best_float32.tflite models/best/best_int8.tflite
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from bovid import config
from bovid.crop import crop_animal, get_detector
from bovid.dataset import labelled_images
from bovid.logging_conf import setup_logging
from bovid.predict import load_model
from bovid.utils import bgr_to_pil, load_image_bgr

logger = logging.getLogger(__name__)


def prepare_crops(images):
    """Crop every image once, so all artifacts are scored on byte-identical inputs."""
    detector = get_detector()
    return [(crop_animal(load_image_bgr(path), detector)[0], breed) for path, breed in images]


def _artifact_size_mb(path):
    """Total on-disk size. ONNX splits weights into a sibling .onnx.data file, which must be
    counted -- otherwise the ONNX artifact looks half its real deployed size."""
    total = os.path.getsize(path)
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        total += os.path.getsize(sidecar)
    return total / 1e6


def _make_runner(path, backend):
    """Return `crop_bgr -> [(breed, conf), ...]` for one artifact.

    Two backends exist because they are not interchangeable: Ultralytics cannot load the INT8
    model at all (its default XNNPACK delegate fails to prepare the quantised graph), while the
    direct TFLite runner is torch-free but reimplements preprocessing. Running FP32 through BOTH
    is what proves the runner's preprocessing matches -- otherwise a preprocessing bug would be
    indistinguishable from quantisation loss.
    """
    if backend == "tflite":
        from bovid.tflite_backend import TFLiteClassifier

        classifier = TFLiteClassifier(path)
        return lambda crop_bgr: classifier.top_k(crop_bgr, k=3)

    model = load_model(path)

    def run(crop_bgr):
        r = model.predict(source=bgr_to_pil(crop_bgr), imgsz=config.DEFAULT_IMGSZ, verbose=False)[0]
        probs = (r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu")
                 else np.array(r.probs.data))
        order = probs.argsort()[::-1][:3]
        return [(r.names[int(i)], float(probs[i])) for i in order]

    return run


def evaluate_model(path, crops, backend):
    runner = _make_runner(path, backend)
    top1_hits = top3_hits = 0
    latencies, records = [], []

    for crop_bgr, true_breed in crops:
        t0 = time.perf_counter()
        predictions = runner(crop_bgr)
        latencies.append((time.perf_counter() - t0) * 1000)

        top3 = [breed for breed, _ in predictions]
        top1_hits += top3[0] == true_breed
        top3_hits += true_breed in top3
        records.append(dict(true_breed=true_breed, top1=top3[0],
                            top1_conf=float(predictions[0][1])))

    n = len(crops)
    return records, dict(
        model=f"{os.path.basename(path)} [{backend}]",
        size_mb=round(_artifact_size_mb(path), 1),
        top1=round(top1_hits / n, 3),
        top3=round(top3_hits / n, 3),
        ms_per_image=round(float(np.median(latencies)), 1),
        mean_conf=round(float(np.mean([r["top1_conf"] for r in records])), 3),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Artifact paths; the FIRST is the baseline others are compared to. "
                             "Suffix a path with ':tflite' to force the direct TFLite runner "
                             "(e.g. best_float32.tflite:tflite), which is how FP32 gets scored "
                             "through both backends to prove they agree.")
    parser.add_argument("--splits", nargs="+", default=["test", "valid"])
    parser.add_argument("--out", default=None, help="Write the summary as JSON here.")
    args = parser.parse_args()

    setup_logging()
    all_summaries = {}

    for split in args.splits:
        images = labelled_images(split)
        logger.info("split %r: cropping %d images once for all artifacts", split, len(images))
        crops = prepare_crops(images)

        baseline, summaries = None, []
        for spec in args.models:
            path, _, forced = spec.partition(":")
            # INT8 must use the direct runner: Ultralytics' XNNPACK delegate cannot load it.
            backend = forced or ("tflite" if "int8" in os.path.basename(path) else "ultralytics")
            logger.info("evaluating %s via %s on %r", path, backend, split)
            records, summary = evaluate_model(path, crops, backend)
            if baseline is None:
                baseline = records
                summary["agree_with_baseline"] = 1.0
            else:
                summary["agree_with_baseline"] = round(
                    np.mean([a["top1"] == b["top1"] for a, b in zip(records, baseline, strict=True)]), 3)
            summaries.append(summary)

        print(f"\n=== {split} (n={len(crops)}) ===")
        print(pd.DataFrame(summaries).to_string(index=False))
        all_summaries[split] = summaries

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
