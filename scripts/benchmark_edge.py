"""Benchmark the torch-free edge pipeline: install size, cold start (measured in a fresh
subprocess -- it only happens once per process), per-stage steady-state latency (median + p95),
and peak RSS. Thread counts are swept because a device has fewer cores than a dev machine.

    python scripts/benchmark_edge.py --n 30 --threads 1 2 4
"""

import argparse
import json
import logging
import os
import resource
import statistics
import subprocess
import sys
import time

from bovid import config
from bovid.logging_conf import setup_logging

logger = logging.getLogger(__name__)

COLD_START_SNIPPET = """
import time, sys
t0 = time.perf_counter()
from bovid.edge import EdgePipeline
t_import = time.perf_counter() - t0

t0 = time.perf_counter()
pipeline = EdgePipeline(num_threads={threads})
t_load = time.perf_counter() - t0

t0 = time.perf_counter()
pipeline.predict({image!r})
t_first = time.perf_counter() - t0

print('COLDSTART', t_import, t_load, t_first)
"""


def sample_images(n):
    test_dir = os.path.join(config.RAW_DATA_DIR, "test")
    paths = sorted(os.path.join(test_dir, p) for p in os.listdir(test_dir) if p.endswith(".jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg images in {test_dir}")
    # Cycle if fewer images than requested, so the timing sample size is what was asked for.
    return [paths[i % len(paths)] for i in range(n)]


def artifact_sizes():
    """On-disk size of everything the edge build must ship."""
    classifier = os.path.join(config.BEST_MODEL_DIR, "best_float32.tflite")
    artifacts = {
        "detector (yolov8n_float32.tflite)": config.EDGE_DETECTOR_MODEL,
        "classifier (best_float32.tflite)": classifier,
        "class names (yolov8n_names.yaml)": os.path.join(
            os.path.dirname(config.EDGE_DETECTOR_MODEL), "yolov8n_names.yaml"),
        "classifier metadata (metadata.yaml)": os.path.join(
            config.BEST_MODEL_DIR, "best_saved_model", "metadata.yaml"),
    }
    sizes = {name: os.path.getsize(path) / 1e6
             for name, path in artifacts.items() if os.path.exists(path)}
    sizes["TOTAL"] = sum(sizes.values())
    return sizes


def measure_cold_start(image, threads):
    """Time-to-first-prediction in a fresh process: import + interpreter build + first infer."""
    code = COLD_START_SNIPPET.format(image=image, threads=threads)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=config.PROJECT_ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"cold-start subprocess failed:\n{out.stderr}")

    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("COLDSTART"))
    _, t_import, t_load, t_first = line.split()
    return dict(import_s=float(t_import), load_models_s=float(t_load),
                first_predict_s=float(t_first),
                total_s=float(t_import) + float(t_load) + float(t_first))


def measure_steady_state(images, threads):
    """Per-stage latency once warm, so the slow stage is identifiable rather than guessed at."""
    from bovid.crop import crop_from_box
    from bovid.edge import EdgePipeline
    from bovid.utils import load_image_bgr

    pipeline = EdgePipeline(num_threads=threads)
    pipeline.predict(images[0])   # warm up: exclude one-off allocation from the steady-state stats

    detect_ms, crop_ms, classify_ms, total_ms = [], [], [], []
    for path in images:
        t_start = time.perf_counter()
        img_bgr = load_image_bgr(path)

        t0 = time.perf_counter()
        box = pipeline.detector.best_animal_box(img_bgr)
        detect_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        crop_bgr, _ = crop_from_box(img_bgr, box)
        crop_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        pipeline.classifier.top_k(crop_bgr, k=config.DEFAULT_TOPK)
        classify_ms.append((time.perf_counter() - t0) * 1000)

        total_ms.append((time.perf_counter() - t_start) * 1000)

    def stats(values):
        ordered = sorted(values)
        return dict(median=round(statistics.median(values), 1),
                    p95=round(ordered[int(0.95 * (len(ordered) - 1))], 1))

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024)   # macOS reports bytes, Linux kilobytes

    return dict(detect=stats(detect_ms), crop=stats(crop_ms), classify=stats(classify_ms),
                total=stats(total_ms), peak_rss_mb=round(peak_rss_mb, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30, help="images to time")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4],
                        help="interpreter thread counts to sweep (1 ~ a low-power core)")
    parser.add_argument("--out", default="runs/edge_benchmark.json")
    args = parser.parse_args()

    setup_logging()
    images = sample_images(args.n)

    sizes = artifact_sizes()
    print("\n=== Edge artifact sizes ===")
    for name, mb in sizes.items():
        print(f"  {name:42s} {mb:7.1f} MB")

    report = {"sizes_mb": {k: round(v, 2) for k, v in sizes.items()}, "runs": {}}

    for threads in args.threads:
        logger.info("benchmarking with num_threads=%d", threads)
        cold = measure_cold_start(images[0], threads)
        steady = measure_steady_state(images, threads)
        report["runs"][threads] = {"cold_start": cold, "steady_state": steady}

        print(f"\n=== num_threads={threads} (n={len(images)}) ===")
        print(f"  cold start   : {cold['total_s']:.2f} s total "
              f"(import {cold['import_s']:.2f}s, load {cold['load_models_s']:.2f}s, "
              f"first predict {cold['first_predict_s']:.2f}s)")
        print(f"  detect       : {steady['detect']['median']:7.1f} ms  "
              f"(p95 {steady['detect']['p95']:.1f})")
        print(f"  crop         : {steady['crop']['median']:7.1f} ms  "
              f"(p95 {steady['crop']['p95']:.1f})")
        print(f"  classify     : {steady['classify']['median']:7.1f} ms  "
              f"(p95 {steady['classify']['p95']:.1f})")
        print(f"  TOTAL/image  : {steady['total']['median']:7.1f} ms  "
              f"(p95 {steady['total']['p95']:.1f})")
        print(f"  peak RSS     : {steady['peak_rss_mb']:7.1f} MB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
