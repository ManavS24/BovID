# Model Card — bovid Indian Bovine Breed Classifier

Following the [Model Cards](https://arxiv.org/abs/1810.03993) format. This documents what the
model is, what it should and should not be used for, how it was evaluated, and its known
limitations — honestly, including the failure modes.

## Model details

- **Task:** single-image classification of Indian bovine (cattle/buffalo) breed — 22 classes.
- **Architecture:** two-stage pipeline. Stage 1 crops the animal (a COCO-pretrained detector with
  a deterministic fallback); stage 2 is a **YOLOv8s-cls** classifier (**5.1M parameters**) over
  the crop. The crop stage is shared code between training and inference, so the model always sees
  the distribution it was trained on.
- **Input:** one RGB photograph (any size; EXIF-orientation-corrected). **Output:** top-3 breeds
  with confidence, plus an explicit *uncertain* flag when top-1 is below a measured threshold.
- **Deployment forms:** server (PyTorch, `best.pt`) and **edge** (FP32 TFLite, torch-free) — the
  two produce the same result contract.
- **Framework / version:** Ultralytics `8.3.195` (pinned; the version the weights were trained and
  exported with).
- **License / intended distribution:** research and educational; an assistive field tool.

### The 22 breeds

Alambadi, Amritmahal, Ayrshire, Banni, Brown Swiss, Gir, Guernsey, Hallikar, Hariana, Holstein
Friesian, Jersey, Kangayam, Kankrej, Krishna Valley, Nagori, Ongole, Pulikulam, Rathi, Red Sindhi,
Sahiwal, Tharparkar, Vechur.

## Intended use

- **Primary use:** an *assistive* first guess at bovine breed from a photo, for field workers,
  extension officers, and farmers — usable **offline on a phone or low-power device**, where
  connectivity is unreliable.
- **Intended users:** people who can treat the output as a ranked suggestion with a confidence,
  and who cross-check uncertain results.
- **Out of scope:**
  - Any **authoritative** or legal/registry breed determination.
  - Breeds **outside the 22** it was trained on — it will still return one of the 22, confidently
    and wrongly.
  - Non-bovine animals, health/age/pregnancy assessment, individual-animal identification.

## Factors

- **Breed similarity.** Several zebu breeds are genuinely close in appearance (Gir↔Banni,
  Kankrej↔Kangayam↔Alambadi); errors concentrate among visually similar breeds, and top-3 is much
  stronger than top-1 for exactly this reason.
- **Capture conditions.** Real farmer photos vary in lighting, angle, occlusion, and background.
  Training augments for watermarks, motion blur, JPEG artifacts and brightness to reflect this,
  but extreme conditions still degrade accuracy.
- **Class balance.** ~32 images/breed; minority-class performance is less reliable.

## Metrics

**Primary metric: grouped 5-fold cross-validation over all 795 images** (not a single split).

| Metric | Score |
|---|---|
| Top-1 accuracy | **53.5% ± 4.5%** |
| Top-3 accuracy | **78.2% ± 2.9%** |

- Chance across 22 breeds is 4.5%; top-1 is ~12× chance.
- **Why grouped CV.** The dataset is Roboflow-augmented: 795 files are only ~328 unique source
  photos (~3 near-identical copies each). A naive random split scatters those copies across
  train/val and **leaks**, reporting a fake **~93%**. Grouping by source photo prevents this — the
  ~53% is the real number. This is the single most important evaluation decision in the project.
- The ±std is the fold-to-fold noise band; a change counts as real only if it moves the mean
  beyond it.

**Edge vs. server:** the torch-free FP32-TFLite edge pipeline matches the server path (51.5% /
42.4% top-1 on the two held-out splits, 82–89% top-1 agreement) — **no accuracy lost** for the
21×-smaller deployment.

## Training data

- **Source:** a Roboflow-exported dataset of 22 Indian bovine breeds, `datasets/indianbovine1/`
  (train/valid/test = 696/66/33 images, 795 total; ~328 unique source photos before augmentation).
- **Labels:** one-hot per image (`_classes.csv` per split).
- **Known data-quality issues (measured, see [METHODOLOGY.md](METHODOLOGY.md) §3):** baked-in
  watermarks/captions on a meaningful fraction of images, ~90% located at the edges/bottom — which
  is *why* the crop stage trims those margins. Contamination is mitigated, not eliminated.

## Evaluation data

Grouped 5-fold CV uses all 795 images (each held out in exactly one fold). Point metrics
(edge/server, per-breed report) additionally use the original held-out `valid` (66) and `test`
(33) splits. The original Roboflow split has **0 source groups spanning splits**, so the fixed
test number is itself un-leaked.

## Ethical considerations & risks

- **Wrong-but-confident outputs.** The model can be confidently wrong, especially on the 22↔lookalike
  breeds or on an out-of-distribution animal. The uncertainty flag mitigates but does not remove
  this; downstream use should never treat a single top-1 as ground truth.
- **Consequential decisions.** Breed can affect valuation, breeding, and subsidy decisions. This
  tool is an aid, not a basis for such decisions on its own.
- **Dataset provenance.** Web-scraped images carry watermarks and unknown collection bias; the 22
  breeds are a subset of India's cattle/buffalo diversity, not the whole.

## Caveats & recommendations

- Treat output as **ranked suggestions with confidence**; prefer **top-3** and heed the uncertain
  flag.
- Re-validate the confidence threshold and CV metrics after any retrain (`bovid evaluate`,
  `scripts/kfold.py`).
- The accuracy ceiling here is **data scarcity + genuine breed similarity**, established by
  measurement — more/cleaner images per breed is the highest-leverage improvement, not a bigger
  model.

---

# API reference

The stable public surface. Full behaviour is in each module's docstring; this is the quick map.

## `bovid.predict` — server inference

```python
load_model(model_path=config.DEFAULT_MODEL_PATH) -> YOLO
    # Load classifier weights (.pt / .onnx / .tflite). task="classify" is set explicitly.

predict(image_path, model=None, imgsz=224, topk=3) -> PredictionResult
    # Full pipeline: crop (shared) -> classify. Loads the model if not given.

predict_batch(image_paths, model=None, imgsz=224, topk=3) -> list[PredictionResult]
    # Sequential by design (batched inference shifts crops near the confidence gate).
```

## `bovid.edge` — torch-free edge inference

```python
EdgePipeline(detector_path=None, classifier_path=None, num_threads=None)
    # TFLite detector + classifier; no torch, no ultralytics. Build once, reuse.
    .predict(image_path, topk=None) -> PredictionResult       # same contract as predict()
    .predict_batch(image_paths, topk=None) -> list[PredictionResult]
```

## `bovid.result` — the shared output contract

```python
Prediction(NamedTuple): breed: str; confidence: float      # unpacks as (breed, confidence)

PredictionResult(frozen dataclass):
    image_path: str
    predictions: list[Prediction]     # highest confidence first
    crop_method: str                  # "detector" | "fallback"
    crop_bgr: ndarray                 # the exact crop fed to the classifier
    .top -> Prediction                # predictions[0]
    .is_confident -> bool             # top-1 >= config.CONFIDENCE_THRESHOLD
```

## `bovid.crop` — the shared crop stage

```python
crop_animal(img_bgr, detector=None) -> (crop_bgr, method)   # single-image (inference) path
select_animal_box(candidates, w, h) -> box | None           # shared gate (server + edge)
```

## `bovid.config` — configuration

Every constant is overridable via a `BOVID_<NAME>` environment variable; values are validated at
import. Key names: `MODELS_ROOT`, `DATA_ROOT`, `DETECTOR_MODEL`, `EDGE_DETECTOR_MODEL`,
`DEFAULT_MODEL_PATH`, `DEVICE` (lazy — resolving it is what imports torch), `CONFIDENCE_THRESHOLD`,
`DEFAULT_TOPK`, `DEFAULT_IMGSZ`. See [INSTALL.md](INSTALL.md) and [USAGE.md](USAGE.md) for the
common overrides.

## Command line

```
bovid predict [--edge] [--model M] [--topk K] [--imgsz N] IMAGE...
bovid evaluate
bovid export [--model M] [--formats ...] [--imgsz N] [--out-dir D]
bovid audit
bovid -v ...            # debug logging
```

Programmatic modules (`train`, `dataset`, `augment`, `export`, `evaluate`, `audit`) are documented
by their module docstrings and driven via the CLI or `scripts/`.
