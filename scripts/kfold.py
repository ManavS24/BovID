"""Stratified-grouped k-fold CV over all 795 images (each held out in one fold), the trustworthy
metric vs the noisy 33-image split. Cropping is done once outside the loop (deterministic, no
leak); oversampling/augmentation happen inside each fold on its train portion only. Reports
mean +/- std top-1/top-3. See METHODOLOGY §5.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from ultralytics import YOLO

from bovid import config
from bovid.augment import build_realworld_train, build_watermarked_train
from bovid.dataset import oversample_train, source_group
from bovid.logging_conf import setup_logging

logger = logging.getLogger(__name__)




def pool_cropped_images(crops_root):
    """All clean cropped images across train/valid/test, as (path, breed, group). Skips any
    oversample dups / watermarked variants (those are per-fold, added inside the loop)."""
    items = []
    for split in ["train", "valid", "test"]:
        split_dir = Path(crops_root) / split
        if not split_dir.is_dir():
            continue
        for breed_dir in split_dir.iterdir():
            if not breed_dir.is_dir():
                continue
            for f in breed_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith(("dup_", "wm_", "aug_")):
                    items.append((f, breed_dir.name, source_group(f.name)))
    return items


def _assemble_fold(items, idx, dst_split_dir):
    for i in idx:
        src, breed, _ = items[i]
        out_dir = os.path.join(dst_split_dir, breed)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(out_dir, src.name))


def _eval_fold(model, val_dir, imgsz, topk=3):
    """Evaluate the held-out fold, reporting plain and TTA (horizontal-flip averaging) in one
    pass so TTA's effect is measured for free."""
    counts = {"plain": [0, 0], "tta": [0, 0]}  # [top1, top3]
    records = []  # out-of-fold per-image predictions, for the label-noise audit
    n = 0
    for breed_dir in Path(val_dir).iterdir():
        if not breed_dir.is_dir():
            continue
        true = breed_dir.name
        for f in breed_dir.iterdir():
            n += 1
            r = model.predict(source=str(f), imgsz=imgsz, device=config.DEVICE, verbose=False)[0]
            base = r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu") else np.array(r.probs.data)
            flipped = Image.open(f).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            r2 = model.predict(source=flipped, imgsz=imgsz, device=config.DEVICE, verbose=False)[0]
            fb = r2.probs.data.cpu().numpy() if hasattr(r2.probs.data, "cpu") else np.array(r2.probs.data)
            for mode, probs in (("plain", base), ("tta", (base + fb) / 2)):
                order = probs.argsort()[::-1][:topk]
                names = [r.names[int(i)] for i in order]
                counts[mode][0] += names[0] == true
                counts[mode][1] += true in names
                if mode == "plain":  # record the adopted (non-TTA) prediction
                    records.append({
                        "file": f.name, "true": true, "pred": names[0],
                        "conf": float(probs[order[0]]),
                        "true_conf": float(probs[list(r.names.values()).index(true)])
                        if true in r.names.values() else 0.0,
                        "top3": names,
                    })
    return {m: (c[0] / n, c[1] / n) for m, c in counts.items()}, n, records


def run_cv(n_folds=5, epochs=40, patience=25, imgsz=224, do_stage2=False,
           watermark_aug=True, richer_aug=False, oversample=True, seed=config.RANDOM_SEED,
           results_path=None, resume=True):
    """Each fold is checkpointed to results_path on completion; a rerun with the same config
    resumes, so an interruption doesn't discard the ~45min already spent."""
    results_path = results_path or os.path.join(config.PROJECT_ROOT, "cv_results.json")
    sig = {"n_folds": n_folds, "epochs": epochs, "patience": patience, "imgsz": imgsz,
           "do_stage2": do_stage2, "watermark_aug": watermark_aug, "richer_aug": richer_aug,
           "oversample": oversample, "seed": seed}
    state = {"config": sig, "folds": {}}
    if resume and os.path.exists(results_path):
        saved = json.load(open(results_path))
        if saved.get("config") == sig:
            state = saved
            logger.info("resuming: %d fold(s) already complete -> %s", len(state["folds"]), sorted(state["folds"]))
        else:
            logger.warning("saved results have a different config -> starting fresh")

    items = pool_cropped_images(config.CROPS_ROOT)
    labels = [breed for _, breed, _ in items]
    groups = [g for _, _, g in items]
    print(f"Pooled {len(items)} cropped images across {len(set(labels))} breeds, "
          f"{len(set(groups))} source groups")

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(items)), labels, groups), 1):
        if str(fold) in state["folds"]:
            logger.info("fold %d/%d already complete, skipping", fold, n_folds)
            continue
        work = tempfile.mkdtemp(prefix=f"cv_fold{fold}_")
        train_dir, val_dir = os.path.join(work, "train"), os.path.join(work, "val")
        _assemble_fold(items, train_idx, train_dir)
        _assemble_fold(items, val_idx, val_dir)
        if oversample:
            oversample_train(train_dir)
        if richer_aug:
            build_realworld_train(train_dir, seed=seed)
        elif watermark_aug:
            build_watermarked_train(train_dir, seed=seed)

        print(f"\n===== Fold {fold}/{n_folds}: train={len(train_idx)} val={len(val_idx)} =====", flush=True)
        model = YOLO(config.CLASSIFIER_MODEL)
        r = model.train(data=work, epochs=epochs, patience=patience, imgsz=imgsz, batch=32,
                        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.01,
                        auto_augment="randaugment", mixup=0.2, cutmix=0.2, erasing=0.5, fliplr=0.5,
                        seed=seed, cache=True, device=config.DEVICE, verbose=False, plots=False)
        best = os.path.join(str(r.save_dir), "weights", "best.pt")
        if do_stage2:
            m2 = YOLO(best)
            r2 = m2.train(data=work, epochs=20, patience=15, imgsz=320, batch=32,
                          optimizer="AdamW", lr0=5e-4, lrf=0.01, weight_decay=0.01,
                          mixup=0.1, cutmix=0.1, erasing=0.4, fliplr=0.5,
                          seed=seed, cache=True, device=config.DEVICE, verbose=False, plots=False)
            best = os.path.join(str(r2.save_dir), "weights", "best.pt")

        res, n, records = _eval_fold(YOLO(best), val_dir, imgsz)
        # checkpoint this fold BEFORE cleanup, so an interruption never loses completed work
        state["folds"][str(fold)] = {"plain": list(res["plain"]), "tta": list(res["tta"]), "n": n}
        state.setdefault("predictions", []).extend(records)
        with open(results_path, "w") as fh:
            json.dump(state, fh, indent=2)
        print(f"Fold {fold}: plain t1={res['plain'][0]:.1%} t3={res['plain'][1]:.1%} | "
              f"tta t1={res['tta'][0]:.1%} t3={res['tta'][1]:.1%} (n={n})  [saved]", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(str(Path(best).parent.parent), ignore_errors=True)  # clean run dir

    folds_done = sorted(state["folds"], key=int)
    print("\n==================== CV SUMMARY ====================")
    print(f"folds={len(folds_done)}/{n_folds} epochs={epochs} stage2={do_stage2} "
          f"richer_aug={richer_aug} watermark_aug={watermark_aug}")
    metrics = {"plain": {"top1": [], "top3": []}, "tta": {"top1": [], "top3": []}}
    for k in folds_done:
        for mode in ("plain", "tta"):
            metrics[mode]["top1"].append(state["folds"][k][mode][0])
            metrics[mode]["top3"].append(state["folds"][k][mode][1])
    for mode in ("plain", "tta"):
        t1, t3 = metrics[mode]["top1"], metrics[mode]["top3"]
        print(f"[{mode:5s}] Top-1: {np.mean(t1):.1%} +/- {np.std(t1):.1%}   {[f'{x:.0%}' for x in t1]}")
        print(f"[{mode:5s}] Top-3: {np.mean(t3):.1%} +/- {np.std(t3):.1%}   {[f'{x:.0%}' for x in t3]}")
    print(f"results: {results_path}")
    return metrics


if __name__ == "__main__":
    setup_logging()
    run_cv()
