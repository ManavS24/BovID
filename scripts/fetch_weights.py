"""Download the stock Ultralytics pretrained weights the pipeline builds on.

These are not committed: yolov8x.pt is 131 MB (over GitHub's 100 MB limit) and all three are
freely downloadable, so they are fetched on setup rather than bloating the repo. The project's
OWN weights (models/best/*, the edge TFLite exports) are tracked -- only these stock bases are
fetched.

    python scripts/fetch_weights.py
"""

import os
import sys
import urllib.request

from bovid import config

# name -> (url, approx size MB for a sanity check)
_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
WEIGHTS = {
    "yolov8x.pt": (f"{_ASSETS}/yolov8x.pt", 131),        # server crop detector
    "yolov8s-cls.pt": (f"{_ASSETS}/yolov8s-cls.pt", 12),  # classifier base (for training)
    "yolov8n.pt": (f"{_ASSETS}/yolov8n.pt", 6),           # source for the edge TFLite detector
}


def _download(url, dest):
    def _progress(block, block_size, total):
        done = block * block_size
        pct = min(100, 100 * done / total) if total > 0 else 0
        sys.stderr.write(f"\r  {os.path.basename(dest)}: {pct:5.1f}%")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, dest, _progress)
    sys.stderr.write("\n")


def main():
    os.makedirs(config.PRETRAINED_DIR, exist_ok=True)
    for name, (url, _) in WEIGHTS.items():
        dest = os.path.join(config.PRETRAINED_DIR, name)
        if os.path.exists(dest):
            print(f"  {name}: already present, skipping")
            continue
        print(f"  {name}: downloading from {url}")
        _download(url, dest)
    print(f"done -> {config.PRETRAINED_DIR}")


if __name__ == "__main__":
    main()
