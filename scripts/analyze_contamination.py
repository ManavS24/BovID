"""Analyze the watermark contamination to decide how to clean it: (1) are flagged images
concentrated in a few breeds (spurious-shortcut risk) or spread out, and (2) do overlays sit at
the edges (crop removes them) or over the animal (needs augmentation)? Rates are lower bounds
(OCR undercounts); the patterns are what matter.
"""

import csv
import os
from collections import defaultdict

import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import Output

from bovid import config
from bovid.logging_conf import setup_logging

AUDIT_CSV = os.path.join(config.PROJECT_ROOT, "docs", "watermark_audit.csv")
EDGE_MARGIN = 0.15   # a word whose center is within the outer 15% on any side counts as "edge"
OCR_CONF_MIN = 40    # ignore low-confidence OCR words when locating overlays
CAPTION_KEYWORDS = ("subscribe", "channel", "share", "youtube", "like")


def build_filename_to_breed():
    """Map every image basename -> (split, breed) using each split's _classes.csv."""
    mapping = {}
    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(config.RAW_DATA_DIR, split, "_classes.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        class_cols = df.columns[1:].tolist()
        for _, row in df.iterrows():
            one_hot = row[class_cols].values
            if one_hot.sum() != 1:
                continue
            breed = class_cols[int(one_hot.argmax())]
            mapping[str(row["filename"])] = (split, breed)
    return mapping


def total_counts_per_breed(fname_to_breed):
    counts = defaultdict(int)
    for _, breed in fname_to_breed.values():
        counts[breed] += 1
    return counts


def load_flagged():
    with open(AUDIT_CSV) as f:
        return list(csv.DictReader(f))


def classify_word_location(cx, cy):
    """edge if center within outer margin on any side; also tag which band."""
    at_edge = cx < EDGE_MARGIN or cx > 1 - EDGE_MARGIN or cy < EDGE_MARGIN or cy > 1 - EDGE_MARGIN
    band = "bottom" if cy > 1 - EDGE_MARGIN else "top" if cy < EDGE_MARGIN else "side" if at_edge else "central"
    return at_edge, band


def overlay_location(img_path):
    """Return (edge_area, central_area, bands) of OCR text in this image, area-weighted."""
    img = Image.open(img_path)
    W, H = img.size
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    edge_area = central_area = 0.0
    bands = defaultdict(float)
    for i, text in enumerate(data["text"]):
        if not text.strip() or int(data["conf"][i]) < OCR_CONF_MIN:
            continue
        w, h = data["width"][i], data["height"][i]
        area = (w * h) / (W * H)
        cx = (data["left"][i] + w / 2) / W
        cy = (data["top"][i] + h / 2) / H
        at_edge, band = classify_word_location(cx, cy)
        bands[band] += area
        if at_edge:
            edge_area += area
        else:
            central_area += area
    return edge_area, central_area, bands


def main():
    fname_to_breed = build_filename_to_breed()
    totals = total_counts_per_breed(fname_to_breed)
    flagged = load_flagged()

    # ---- 1) Class-correlation ----
    flagged_per_breed = defaultdict(int)
    caption_breeds = set()
    unmapped = 0
    for row in flagged:
        basename = os.path.basename(row["path"])
        if basename not in fname_to_breed:
            unmapped += 1
            continue
        _, breed = fname_to_breed[basename]
        flagged_per_breed[breed] += 1
        if any(kw in row["ocr_snippet"].lower() for kw in CAPTION_KEYWORDS):
            caption_breeds.add(breed)

    total_flagged = sum(flagged_per_breed.values())
    overall_rate = total_flagged / sum(totals.values())

    print("=" * 68)
    print("1) CLASS-CORRELATION  (OCR-detected flags; lower bound -- see caveat)")
    print("=" * 68)
    print(f"Overall flag rate: {total_flagged}/{sum(totals.values())} = {overall_rate:.1%}\n")
    print(f"{'breed':18s} {'flagged':>8s} {'total':>6s} {'rate':>7s}   concentration")
    for breed in sorted(totals, key=lambda b: -flagged_per_breed[b] / totals[b] if totals[b] else 0):
        flg, tot = flagged_per_breed[breed], totals[breed]
        rate = flg / tot if tot else 0
        marker = "  <-- HIGH" if rate > overall_rate * 2 and flg >= 2 else ""
        if flg:
            print(f"{breed:18s} {flg:>8d} {tot:>6d} {rate:>6.1%}{marker}")
    if unmapped:
        print(f"\n({unmapped} flagged files had no breed mapping -- e.g. Screenshot-*/numeric names)")

    print(f"\nYouTube/'subscribe'-style captions appear across {len(caption_breeds)} distinct "
          f"breed(s): {sorted(caption_breeds)}")
    print("  -> spread across many breeds = noise (harmless); concentrated in 1-2 = shortcut risk")

    # ---- 2) Overlay location ----
    print("\n" + "=" * 68)
    print("2) OVERLAY LOCATION  (area-weighted, OCR-detectable text only)")
    print("=" * 68)
    tot_edge = tot_central = 0.0
    band_totals = defaultdict(float)
    per_image = []
    for row in flagged:
        edge, central, bands = overlay_location(row["path"])
        if edge + central == 0:
            continue
        tot_edge += edge
        tot_central += central
        for b, a in bands.items():
            band_totals[b] += a
        per_image.append((os.path.basename(row["path"]), edge, central))

    grand = tot_edge + tot_central
    if grand:
        print(f"Overlay text area at EDGE (outer {EDGE_MARGIN:.0%}): {tot_edge/grand:.0%}")
        print(f"Overlay text area CENTRAL:                    {tot_central/grand:.0%}")
        print("\nBy band (share of overlay area):")
        for band in ("bottom", "top", "side", "central"):
            if band_totals[band]:
                print(f"  {band:8s}: {band_totals[band]/grand:.0%}")
        edge_dominant = sum(1 for _, e, c in per_image if e > c)
        print(f"\nImages where overlay is mostly at the edge: "
              f"{edge_dominant}/{len(per_image)} "
              f"({edge_dominant/len(per_image):.0%}) -- these a detector-crop should clean")


if __name__ == "__main__":
    setup_logging()
    main()
