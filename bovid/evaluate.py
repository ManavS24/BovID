"""Evaluate the full crop+classify pipeline against the labelled test set: top-1/top-3, a
per-class report, and confidence calibration. Measures the raw-photo-in path a user gets, not
YOLO's training metrics on already-cropped images.
"""

import os

import pandas as pd
from sklearn.metrics import classification_report

from . import config
from .predict import load_model, predict


def topk_metrics(records):
    """Accuracy + confidence-calibration stats from (true_breed, ranked_breeds, top1_conf)
    records. A pure function so the metric arithmetic is unit-testable without a model."""
    n = len(records)
    if n == 0:
        raise ValueError("no records to score")

    top1_hits = top3_hits = 0
    correct_conf, incorrect_conf = [], []
    for true_breed, ranked, top1_conf in records:
        if ranked[0] == true_breed:
            top1_hits += 1
            correct_conf.append(top1_conf)
        else:
            incorrect_conf.append(top1_conf)
        if true_breed in ranked:
            top3_hits += 1

    return {
        "n": n,
        "top1_acc": top1_hits / n,
        "top3_acc": top3_hits / n,
        "correct_conf": correct_conf,
        "incorrect_conf": incorrect_conf,
    }


def run_evaluation(test_dir=None, topk=3):
    test_dir = test_dir or os.path.join(config.RAW_DATA_DIR, "test")
    df = pd.read_csv(os.path.join(test_dir, "_classes.csv"))
    class_cols = df.columns[1:].tolist()

    model = load_model()

    y_true, y_top1, records = [], [], []
    for _, row in df.iterrows():
        fname = str(row["filename"])
        img_path = os.path.join(test_dir, fname)
        if not os.path.exists(img_path):
            continue
        one_hot = row[class_cols].values
        if one_hot.sum() != 1:
            continue
        true_breed = class_cols[int(one_hot.argmax())]

        result = predict(img_path, model=model, topk=topk)
        ranked = [p.breed for p in result.predictions]
        y_true.append(true_breed)
        y_top1.append(ranked[0])
        records.append((true_breed, ranked, result.top.confidence))

    metrics = topk_metrics(records)
    # y_true / y_top1 are kept for the per-class classification_report in main().
    metrics["y_true"] = y_true
    metrics["y_top1"] = y_top1
    return metrics


def main():
    results = run_evaluation()
    n = results["n"]

    print(f"Evaluated {n} test images from {config.RAW_DATA_DIR}/test\n")
    print(f"Top-1 accuracy: {results['top1_acc']:.1%}")
    print(f"Top-3 accuracy: {results['top3_acc']:.1%}\n")

    print("Per-class report:")
    print(classification_report(results["y_true"], results["y_top1"], zero_division=0))

    cc, ic = results["correct_conf"], results["incorrect_conf"]
    print("Confidence calibration (top-1 confidence, correct vs. incorrect):")
    if cc:
        print(f" - mean confidence | correct:   {sum(cc)/len(cc):.3f}  (n={len(cc)})")
    if ic:
        print(f" - mean confidence | incorrect: {sum(ic)/len(ic):.3f}  (n={len(ic)})")
        for threshold in (0.5, 0.7, config.CONFIDENCE_THRESHOLD):
            caught = sum(1 for c in ic if c < threshold)
            false_flags = sum(1 for c in cc if c < threshold)
            print(f" - threshold={threshold}: flags {caught}/{len(ic)} wrong predictions as "
                  f"uncertain, but also flags {false_flags}/{len(cc)} correct ones")


if __name__ == "__main__":
    main()
