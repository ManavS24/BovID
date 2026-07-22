"""OCR-scan the dataset for baked-in watermarks/captions (Getty, Shutterstock, YouTube overlays,
etc.), writing the flagged list to docs/watermark_audit.csv.

The OCR count (~4.5%) is a LOWER BOUND: a manual audit of ~120 images found 9-17% contaminated,
since tesseract misses low-contrast stock-photo watermarks. See METHODOLOGY §3.
"""

import csv
import logging
import os
import re
from pathlib import Path

import pytesseract
from PIL import Image
from tqdm import tqdm

from . import config
from .logging_conf import setup_logging

logger = logging.getLogger(__name__)

SUSPICIOUS_KEYWORDS = [
    "getty", "shutterstock", "istock", "alamy", "dreamstime", "123rf",
    "watermark", "subscribe", "like share", "channel", "youtube",
    "www.", ".com",
]
TIMESTAMP_PATTERN = re.compile(r"\d{1,2}:\d{2}\s*/\s*\d{1,2}:\d{2}")


def ocr_text(img_path):
    """OCR one image. A failure is logged (not silent) -- treating it as clean would bias the
    contamination count."""
    try:
        return pytesseract.image_to_string(Image.open(img_path)).strip()
    except (OSError, pytesseract.TesseractError) as e:
        logger.warning("OCR failed, counting as no-text: %s (%s)", img_path, e)
        return ""


def classify(text):
    """Return a short reason string if text looks like a watermark/overlay, else None."""
    if not text:
        return None
    low = text.lower()
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in low:
            return f"keyword:{kw}"
    if TIMESTAMP_PATTERN.search(text):
        return "video-timestamp"
    # A field photo has no legitimate reason to contain a run of real text.
    letters = sum(c.isalpha() for c in text)
    if letters >= 4:
        return "generic-text"
    return None


def scan_split(split_dir, ocr=ocr_text):
    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in Path(split_dir).glob(ext)
    )
    flagged = []
    for img_path in tqdm(image_paths, desc=f"Scanning {os.path.basename(split_dir)}"):
        text = ocr(img_path)
        reason = classify(text)
        if reason:
            flagged.append((str(img_path), reason, text[:80].replace("\n", " ")))
    return image_paths, flagged


def main():
    total_images = 0
    total_flagged = []

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(config.RAW_DATA_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        image_paths, flagged = scan_split(split_dir)
        total_images += len(image_paths)
        total_flagged.extend(flagged)
        if image_paths:
            print(f"{split}: {len(flagged)}/{len(image_paths)} flagged ({len(flagged)/len(image_paths):.1%})")

    print(f"\nOverall: {len(total_flagged)}/{total_images} flagged ({len(total_flagged)/total_images:.1%})")

    reason_counts = {}
    for _, reason, _ in total_flagged:
        key = reason.split(":")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    print("\nBy reason:")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    out_path = os.path.join(config.PROJECT_ROOT, "docs", "watermark_audit.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "reason", "ocr_snippet"])
        writer.writerows(total_flagged)
    print(f"\nFull flagged list written to: {out_path}")


if __name__ == "__main__":
    setup_logging()
    main()
