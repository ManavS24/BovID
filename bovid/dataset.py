"""Data preparation: crop each source image (via crop.py) into the YOLO-cls train/val/test
layout.
"""

import logging
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .crop import crop_animal, get_detector
from .utils import bgr_to_pil, load_image_bgr

logger = logging.getLogger(__name__)


def labelled_images(split, limit=None, seed=None):
    """(path, breed) pairs for a split (cleanly single-label, on-disk rows only); `limit` samples
    stratified across breeds. Pass a HELD-OUT split for accuracy comparisons -- `train` was
    memorised and reports ~100%, hiding regressions."""
    src_dir = os.path.join(config.RAW_DATA_DIR, split)
    df = pd.read_csv(os.path.join(src_dir, "_classes.csv"))
    class_cols = df.columns[1:].tolist()

    rows = []
    for _, row in df.iterrows():
        one_hot = row[class_cols].values
        if one_hot.sum() != 1:
            continue
        path = os.path.join(src_dir, str(row["filename"]))
        if os.path.exists(path):
            rows.append((path, class_cols[int(np.argmax(one_hot))]))

    if limit is None or limit >= len(rows):
        return rows

    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    by_breed = {}
    for path, breed in rows:
        by_breed.setdefault(breed, []).append((path, breed))

    picked, per_breed = [], max(1, limit // len(by_breed))
    for breed in sorted(by_breed):
        group = by_breed[breed]
        idx = rng.permutation(len(group))[:per_breed]
        picked.extend(group[i] for i in idx)
    return picked[:limit]


def source_group(filename):
    """Group key = everything before '.rf.', so a photo's ~3 Roboflow augmentation siblings stay
    in one CV fold. Without this they straddle train/val and leak (fake ~93% vs real ~50%)."""
    return filename.split(".rf.")[0]


def build_crops(src_base, out_root):
    """Crop each labelled image (via crop_animal) into out_root/{split}/{breed}/, reporting the
    detector-vs-fallback split."""
    detector = get_detector()

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    for split in ["train", "valid", "test"]:
        src_dir = os.path.join(src_base, split)
        csv_path = os.path.join(src_dir, "_classes.csv")
        if not os.path.exists(csv_path):
            logger.warning("missing %s — skipping split %s", csv_path, split)
            continue

        df = pd.read_csv(csv_path)
        class_cols = df.columns[1:].tolist()
        for b in class_cols:
            os.makedirs(os.path.join(out_root, split, b), exist_ok=True)

        copied, missing = 0, 0
        methods = Counter()
        logger.info("preparing split %r from %s (rows=%d)", split, src_dir, len(df))

        for _, row in tqdm(df.iterrows(), total=len(df)):
            fname = str(row["filename"])
            one_hot = row[class_cols].values
            if one_hot.sum() != 1:
                continue
            breed = class_cols[int(np.argmax(one_hot))]
            src_img = os.path.join(src_dir, fname)
            if not os.path.exists(src_img):
                missing += 1
                continue

            # Scraped files may be truncated/undecodable: report and skip, never substitute.
            try:
                img_bgr = load_image_bgr(src_img)
            except OSError as e:
                logger.warning("unreadable image, skipping: %s (%s)", src_img, e)
                missing += 1
                continue

            crop_bgr, method = crop_animal(img_bgr, detector)
            methods[method] += 1
            out_path = os.path.join(out_root, split, breed, os.path.basename(fname))
            bgr_to_pil(crop_bgr).save(out_path, format="JPEG", quality=95)
            copied += 1

        total = sum(methods.values()) or 1
        logger.info("split %s done: copied=%d missing=%d | detector=%d (%.0f%%) fallback=%d (%.0f%%)",
                    split, copied, missing,
                    methods["detector"], 100 * methods["detector"] / total,
                    methods["fallback"], 100 * methods["fallback"] / total)


def prepare_yolo_dataset(crops_root, yolo_data_dir):
    """Copy crops/{train,valid,test} into the train/val/test layout Ultralytics classification expects."""
    if os.path.exists(yolo_data_dir):
        shutil.rmtree(yolo_data_dir)

    for split in ["train", "val", "test"]:
        src_split = "valid" if split == "val" else split
        src_dir = os.path.join(crops_root, src_split)
        dst_dir = os.path.join(yolo_data_dir, split)
        if not os.path.exists(src_dir):
            logger.warning("missing crops split: %s", src_dir)
            continue
        logger.info("copying crops %s -> %s", src_dir, dst_dir)
        shutil.copytree(src_dir, dst_dir)


def oversample_train(dst_train_dir):
    """Duplicate files in minority classes so every class matches the largest class's count."""
    breeds = sorted(p for p in Path(dst_train_dir).iterdir() if p.is_dir())
    counts = {b.name: len(list(b.glob("*"))) for b in breeds}
    if not counts:
        logger.warning("no train images found to oversample")
        return

    max_n = max(counts.values())
    logger.info("train counts before oversample: %s (max=%d)", counts, max_n)
    for b in breeds:
        files = list(b.glob("*"))
        need = max_n - len(files)
        if need <= 0 or not files:
            continue
        for i in range(need):
            src = files[i % len(files)]
            shutil.copy2(src, b / f"dup_{i}_{src.name}")

    counts_after = {b.name: len(list(b.glob("*"))) for b in breeds}
    logger.info("train counts after oversample: %s", counts_after)
