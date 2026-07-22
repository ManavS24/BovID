"""Compare the torch-free edge pipeline against the server pipeline on held-out data: does the
edge device return the same breed, and is it as accurate? (The edge build swaps both models and
reimplements preprocessing/decode/NMS in numpy, any of which could drift.)

    python scripts/compare_edge_server.py --splits test valid
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from bovid.dataset import labelled_images
from bovid.logging_conf import setup_logging

logger = logging.getLogger(__name__)


def run_server(images):
    from bovid.predict import load_model, predict

    model = load_model()
    results, latencies = [], []
    for path, breed in images:
        t0 = time.perf_counter()
        r = predict(path, model=model)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r, breed))
    return results, latencies


def run_edge(images):
    from bovid.edge import EdgePipeline

    pipeline = EdgePipeline()
    results, latencies = [], []
    for path, breed in images:
        t0 = time.perf_counter()
        r = pipeline.predict(path)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r, breed))
    return results, latencies


def summarise(name, results, latencies):
    n = len(results)
    return dict(
        pipeline=name,
        top1=round(sum(r.top.breed == breed for r, breed in results) / n, 3),
        top3=round(sum(breed in [p.breed for p in r.predictions] for r, breed in results) / n, 3),
        detector_crops=round(sum(r.crop_method == "detector" for r, _ in results) / n, 3),
        ms_per_image=round(float(np.median(latencies)), 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["test", "valid"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    all_summaries = {}

    for split in args.splits:
        images = labelled_images(split)
        logger.info("split %r: %d images", split, len(images))

        server, server_ms = run_server(images)
        edge, edge_ms = run_edge(images)

        rows = [summarise("server (torch)", server, server_ms),
                summarise("edge (tflite, torch-free)", edge, edge_ms)]
        agree = np.mean([s.top.breed == e.top.breed for (s, _), (e, _) in zip(server, edge, strict=True)])
        same_crop = np.mean([s.crop_method == e.crop_method for (s, _), (e, _) in zip(server, edge, strict=True)])
        rows[1]["agrees_with_server"] = round(float(agree), 3)
        rows[1]["same_crop_method"] = round(float(same_crop), 3)
        rows[0]["agrees_with_server"] = 1.0
        rows[0]["same_crop_method"] = 1.0

        print(f"\n=== {split} (n={len(images)}) ===")
        print(pd.DataFrame(rows).to_string(index=False))
        all_summaries[split] = rows

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
