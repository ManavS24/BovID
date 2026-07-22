"""Label-noise audit: find probable mislabels from out-of-fold CV predictions (a model can't
reveal a wrong label on images it trained on). Scores source GROUPS, not files -- all ~3 siblings
of a photo confidently predicted as the same other breed is strong evidence the label is wrong.
"""

import csv
import json
import os
from collections import defaultdict

from bovid import config
from bovid.dataset import source_group
from bovid.logging_conf import setup_logging


def load_predictions(results_path):
    with open(results_path) as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    if not preds:
        raise ValueError(f"No per-image predictions in {results_path} — rerun run_cv to collect them.")
    return preds


def audit(results_path=None, min_conf=0.5):
    """Rank source groups by how confidently the model disagrees with their label."""
    results_path = results_path or os.path.join(config.PROJECT_ROOT, "cv_results_step3.json")
    preds = load_predictions(results_path)

    groups = defaultdict(list)
    for r in preds:
        groups[source_group(r["file"])].append(r)

    candidates = []
    for group, recs in groups.items():
        true = recs[0]["true"]
        wrong = [r for r in recs if r["pred"] != true]
        if not wrong:
            continue
        # do all siblings agree on the SAME wrong breed?
        wrong_preds = {r["pred"] for r in wrong}
        unanimous = len(wrong) == len(recs) and len(wrong_preds) == 1
        mean_conf = sum(r["conf"] for r in wrong) / len(wrong)
        mean_true_conf = sum(r["true_conf"] for r in recs) / len(recs)
        candidates.append({
            "group": group,
            "labelled": true,
            "predicted": max(wrong_preds, key=lambda p: sum(1 for r in wrong if r["pred"] == p)),
            "siblings_wrong": f"{len(wrong)}/{len(recs)}",
            "unanimous": unanimous,
            "mean_pred_conf": round(mean_conf, 3),
            "mean_true_conf": round(mean_true_conf, 3),
            # confident disagreement: model sure it's X, and sure it is NOT the label
            "score": round(mean_conf - mean_true_conf, 3),
        })

    # strongest evidence first: unanimous siblings, then confident disagreement
    candidates.sort(key=lambda c: (c["unanimous"], c["score"]), reverse=True)
    strong = [c for c in candidates if c["unanimous"] and c["mean_pred_conf"] >= min_conf]

    print(f"Source groups total: {len(groups)} | groups with any wrong prediction: {len(candidates)}")
    print(f"STRONG mislabel candidates (all siblings agree on same wrong breed, conf>={min_conf}): {len(strong)}\n")
    print(f"{'labelled':18s} {'predicted':18s} {'sibs':6s} {'pconf':6s} {'tconf':6s}  group")
    for c in strong[:30]:
        print(f"{c['labelled']:18s} {c['predicted']:18s} {c['siblings_wrong']:6s} "
              f"{c['mean_pred_conf']:<6} {c['mean_true_conf']:<6}  {c['group'][:44]}")

    out_csv = os.path.join(config.PROJECT_ROOT, "docs", "label_audit.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)
    print(f"\nFull ranked list -> {out_csv}")
    return strong, candidates


if __name__ == "__main__":
    setup_logging()
    audit()
