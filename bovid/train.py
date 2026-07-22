"""Canonical training pipeline: crop dataset -> YOLO-cls layout -> 2-stage train -> validate ->
export. Runs against this repo's datasets/ layout; see config.py to relocate it.
"""

import logging
import os
import random
import shutil

import numpy as np
from ultralytics import YOLO

from . import config
from .augment import build_watermarked_train
from .dataset import build_crops, oversample_train, prepare_yolo_dataset
from .export import export_model
from .logging_conf import setup_logging

logger = logging.getLogger(__name__)

RAW_DATA_DIR = config.RAW_DATA_DIR
CROPS_ROOT = config.CROPS_ROOT
YOLO_DATA_DIR = config.YOLO_DATA_DIR

CLASSIFIER_MODEL = config.CLASSIFIER_MODEL
RUNS_DIR = config.RUNS_DIR
BEST_MODEL_DIR = config.BEST_MODEL_DIR

DO_OVERSAMPLE_TRAIN = config.DO_OVERSAMPLE_TRAIN
DO_FINETUNE_320 = config.DO_FINETUNE_320
DO_WATERMARK_AUG = config.DO_WATERMARK_AUG
RANDOM_SEED = config.RANDOM_SEED

STAGE1 = config.STAGE1
STAGE2 = config.STAGE2


def check_source_splits(src_base):
    for split in ["train", "valid", "test"]:
        d = os.path.join(src_base, split)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"CRITICAL: Required data directory not found: {d}")


def best_weights_of(train_result):
    """Best checkpoint of a finished run, from results.save_dir (ask Ultralytics rather than
    guess where it wrote)."""
    best = os.path.join(str(train_result.save_dir), "weights", "best.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"training finished but no weights at {best}")
    return best


def mb(path):
    """Size in MB, or None if the artifact wasn't produced (e.g. an export that was skipped)."""
    if not path or not os.path.exists(str(path)):
        return None
    return round(os.path.getsize(str(path)) / 1024 / 1024, 3)


def count_params(best_pt):
    """Total parameters, summed directly from the loaded torch model."""
    return sum(p.numel() for p in YOLO(best_pt).model.parameters())


def main():
    check_source_splits(RAW_DATA_DIR)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1) crop dataset -> YOLO-cls layout
    build_crops(RAW_DATA_DIR, CROPS_ROOT)
    prepare_yolo_dataset(CROPS_ROOT, YOLO_DATA_DIR)
    if DO_OVERSAMPLE_TRAIN:
        oversample_train(os.path.join(YOLO_DATA_DIR, "train"))
    if DO_WATERMARK_AUG:
        build_watermarked_train(os.path.join(YOLO_DATA_DIR, "train"), seed=RANDOM_SEED)

    os.system(f"yolo settings runs_dir={RUNS_DIR}")

    # 2) stage 1 training (224px)
    logger.info("stage 1 training (224px)")
    model = YOLO(CLASSIFIER_MODEL)
    result = model.train(
        data=YOLO_DATA_DIR, optimizer="AdamW", seed=RANDOM_SEED, cache=True, device=config.DEVICE,
        auto_augment="randaugment", mixup=0.2, cutmix=0.2, erasing=0.5, fliplr=0.5,
        lrf=0.01, weight_decay=0.01, **STAGE1,
    )
    best_pt = best_weights_of(result)

    # 3) stage 2 fine-tune (320px)
    if DO_FINETUNE_320:
        logger.info("stage 2 fine-tune (320px)")
        model2 = YOLO(best_pt)
        result2 = model2.train(
            data=YOLO_DATA_DIR, optimizer="AdamW", seed=RANDOM_SEED, cache=True, device=config.DEVICE,
            mixup=0.1, cutmix=0.1, erasing=0.4, fliplr=0.5,
            lrf=0.01, weight_decay=0.01, **STAGE2,
        )
        best_pt = best_weights_of(result2)

    persisted_best_pt = os.path.join(BEST_MODEL_DIR, "best.pt")
    shutil.copy2(best_pt, persisted_best_pt)
    logger.info("copied final best.pt to %s", persisted_best_pt)

    # 4) validate / test
    m_eval = YOLO(persisted_best_pt)
    val_metrics = m_eval.val(data=YOLO_DATA_DIR, imgsz=STAGE1["imgsz"], device=config.DEVICE)
    test_metrics = m_eval.val(data=YOLO_DATA_DIR, split="test", imgsz=STAGE1["imgsz"], device=config.DEVICE)

    # 5) export deployment formats (see export.py)
    persisted = export_model(persisted_best_pt, imgsz=STAGE1["imgsz"], out_dir=BEST_MODEL_DIR)

    # 6) final summary
    params = count_params(persisted_best_pt)

    print("\n==================== FINAL SUMMARY ====================")
    print("Persisted best.pt path:", persisted_best_pt)
    for name, path in persisted.items():
        print(f"Persisted {name} path:", path)

    print("\nModel Sizes (MB):")
    print(" - best.pt:", mb(persisted_best_pt))
    for name, path in persisted.items():
        print(f" - {name}:", mb(path))

    print("\nAccuracy:")
    print(" - Val Top1:", val_metrics.results_dict.get("metrics/accuracy_top1"))
    print(" - Val Top5:", val_metrics.results_dict.get("metrics/accuracy_top5"))
    print(" - Test Top1:", test_metrics.results_dict.get("metrics/accuracy_top1"))
    print(" - Test Top5:", test_metrics.results_dict.get("metrics/accuracy_top5"))

    print(f"\nParameters: {params:,}")
    print("\n==================== DONE ====================")


if __name__ == "__main__":
    setup_logging()
    main()
