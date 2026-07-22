# Methodology & Architecture

Indian bovine (cattle/buffalo) breed classification from a single photograph, built to run
offline on farmer-facing edge devices. This document records the design, the data-quality
findings, the evaluation methodology, and the key engineering decisions.

> Status note: the model-improvement sequence (k-fold CV → real-world augmentation + TTA →
> label-noise audit) is in progress. Numbers marked *(baseline)* are final; the final selected
> model's numbers and exports are filled in once that sequence completes.

---

## 1. Problem & scope

- **Task:** given one image, predict the Indian bovine breed (22 classes: Alambadi, Amritmahal,
  Ayrshire, Banni, Brown_Swiss, Gir, Guernsey, Hallikar, Hariana, Holstein_Friesian, Jersey,
  Kangayam, Kankrej, Krishna_Valley, Nagori, Ongole, Pulikulam, Rathi, Red_Sindhi, Sahiwal,
  Tharparkar, Vechur).
- **Deployment goal:** run **offline on edge devices** (a farmer's phone / low-power hardware),
  because target users often lack reliable connectivity.
- **Non-goals:** it is an assistive classifier, not an authority; predictions are surfaced with
  confidence and an explicit uncertainty signal.

## 2. Architecture: detect → crop → classify

A two-stage pipeline, identical at training and inference time (the crop is shared code, so the
model always sees the same distribution it was trained on):

```
raw image ──▶ [1] animal detector ──▶ crop to animal ──▶ [2] breed classifier ──▶ top-3 breeds
                    │  (fallback if no confident detection)         │
                    └──▶ deterministic bottom/edge-trim + center ───┘
```

**Stage 1 — hybrid crop.** A COCO-pretrained detector locates the animal and we crop tightly to
it (with asymmetric padding — no padding *below* the animal, so ground-level watermarks are
excluded). When the detector finds nothing confident enough (≈15% of train / ≈6% of val-test), we
fall back to a deterministic trim of the bottom/edge margins followed by a centre-square crop.
This removes most baked-in overlays for free and tightens the frame to the discriminative region.

**Stage 2 — classifier.** A YOLOv8s-cls model (≈5M params) predicts the breed over the crop and
returns the top-3 with confidence. A measured confidence threshold flags low-confidence outputs as
uncertain rather than asserting them.

### Server vs. edge split
The server/desktop path uses a large, accurate detector (yolov8x, 137MB, PyTorch). That is
**infeasible on edge**, so the edge path uses a **tiny TFLite detector** (yolov8n → TFLite) plus an
**INT8-quantised TFLite classifier**, run through a **torch-free** runtime. Both paths share the
same crop→classify logic and result contract; only the model backends differ. Selection is
config-only — no code change:

```bash
BOVID_DETECTOR_MODEL=models/pretrained/yolov8n_float32.tflite bovid predict img.jpg
```

#### Choosing the edge detector (measured, and it overturned the assumption)

The obvious way to compare detectors is detection rate, and by that measure the tiny model looks
disqualifying: yolov8n finds an animal in 56–70% of images versus yolov8x's 94%. **That metric is
misleading here.** A missed detection is not an error — it falls through to the deterministic
margin trim, which is a usable crop. What matters is breed accuracy, so
`scripts/compare_detectors.py` measures exactly that, on **held-out** data:

| Split | Detector | Size | Detect rate | Top-1 | Top-3 | Detect ms |
|---|---|---|---|---|---|---|
| test (n=33) | yolov8x.pt | 136.9 MB | 93.9% | 48.5% | 84.8% | 148.6 |
| test | yolov8n.pt | 6.5 MB | 69.7% | 51.5% | 84.8% | 24.4 |
| test | **yolov8n_float32.tflite** | 12.9 MB | 69.7% | 51.5% | 84.8% | 130.1 |
| valid (n=65) | yolov8x.pt | 136.9 MB | 93.9% | 37.9% | 68.2% | 147.7 |
| valid | yolov8n.pt | 6.5 MB | 56.1% | 42.4% | 69.7% | 26.8 |
| valid | **yolov8n_float32.tflite** | 12.9 MB | 56.1% | 42.4% | 69.7% | 131.9 |

Conclusions:

- **The tiny detector costs no measurable accuracy.** It scored equal-or-better top-1 on both
  held-out splits. The gaps favour yolov8n, but at n=33/65 a few images swing several points, so
  the honest claim is *no detectable penalty* — not *better*.
- **The TFLite export is faithful.** It reproduces the `.pt` detector's detection rate, top-1,
  top-3 and crop-box IoU **exactly** — so conversion introduced no drift.
- **Crops largely agree** with the yolov8x baseline: median IoU 0.93 where both detectors fire,
  and identical top-1 breed on 82–89% of images.
- **The TFLite latency (~131 ms) is not the edge number.** It is slower than yolov8n.pt (~25 ms)
  only because torch uses Apple's MPS GPU here while TFLite runs on CPU. On the actual target —
  a phone or Pi — torch is not an option at all. T16 benchmarks the real device.

> **Methodological note.** The first run of this comparison sampled the *train* split and reported
> top-1 = 1.00 for every detector — the classifier had memorised those images. Detector choice
> looked irrelevant because the metric was saturated, not because the detectors were equivalent.
> `--split` now defaults to held-out data so the trap cannot recur silently.

#### Why FP32 and not FP16 TFLite

The FP16 export is half the size and looks like the obvious pick, but **it cannot run on CPU**:
`onnx2tf` gives it a float16 *input tensor*, and TFLite's `CONV_2D` kernel accepts only
fp32/uint8/int8/int16, so `allocate_tensors()` fails with *"Node number 1 (CONV_2D) failed to
prepare"*. FP16 TFLite is a GPU-delegate format; this edge target is CPU. The FP16 file is kept
for a future GPU-delegate path, but FP32 is the default. Real size reduction comes from INT8
quantisation (T14), which is a deliberate step with an accuracy check — not a silent default.

### Edge classifier: INT8 quantisation was measured and REJECTED

The plan was to ship an INT8-quantised classifier (4× smaller). It does not work for this model.
Measured on held-out data, with the crop stage held constant so only the model format varies
(`scripts/compare_classifiers.py`):

| Artifact | Size | test top-1 | test top-3 | valid top-1 | valid top-3 | Agrees with `best.pt` |
|---|---|---|---|---|---|---|
| `best.pt` (baseline) | 10.3 MB | 48.5% | 84.8% | 37.9% | 68.2% | — |
| `best_float32.tflite` | 20.4 MB | **48.5%** | **84.8%** | **37.9%** | **68.2%** | **100%** |
| INT8 full-integer (val-calibrated) | 5.2 MB | 3.0% | 12.1% | 6.1% | 16.7% | 3–8% |
| INT8 full-integer (train-calibrated) | 5.2 MB | 6.1% | 18.2% | 9.1% | 15.2% | 6–9% |
| INT8 dynamic-range | 5.2 MB | 6.1% | 15.2% | 4.5% | 13.6% | 5–6% |

Random guessing across 22 breeds is 4.5%, so **every INT8 variant is at or barely above chance**.
This is not gradual degradation — the full-integer model collapses onto 4 distinct predicted
classes out of 22 (mostly Tharparkar/Vechur), and the dynamic-range model returns NaN
confidences. Three independent routes were tried before concluding:

1. **Ultralytics' default INT8 export** calibrated on the *val* split — 66 images, 3 per breed.
   Ultralytics itself warns ">300 images recommended for INT8 calibration, found 66". That was a
   real bug (`split="train"` in `EXPORT_FORMATS` now calibrates on all 1834 training images)…
2. **…but fixing it barely moved the number** (3.0% → 6.1%), which rules out calibration size as
   the cause.
3. **Dynamic-range quantisation** (weights-only, normally near-lossless) also collapsed, and
   produced NaNs.

Since the FP32 TFLite export is *bit-faithful* to `best.pt` (100% agreement, identical accuracy),
the conversion pipeline is sound — the loss is specific to INT8 quantisation of this
architecture. **Decision: the edge classifier ships as FP32 TFLite.** A 4× size saving is not
worth a 40-point accuracy collapse, and a model at chance level would make the demo worthless.

FP16 was checked as a middle option and **cannot run on CPU** — same failure as the FP16 detector
(fp16 input tensor, rejected by TFLite's `CONV_2D`). The INT8 and FP16 artifacts were deleted
rather than committed, so nobody mistakes them for usable weights.

> Re-testing INT8 is worthwhile if the model is ever retrained, ideally with
> quantisation-aware training or per-channel quantisation, but it must clear the same held-out
> accuracy bar before shipping.

### The torch-free edge path (measured against the server path)

`bovid/edge.py` runs the same detect → crop → classify pipeline with **no torch and no
ultralytics**, returning the identical `PredictionResult`, so the CLI, app and tests cannot tell
which backend produced a result:

|  | server (`bovid.predict`) | edge (`bovid.edge`) |
|---|---|---|
| Detector | `yolov8x.pt` — 137 MB, torch | `yolov8n_float32.tflite` — 13 MB |
| Classifier | `best.pt` — 10 MB, torch | `best_float32.tflite` — 20 MB |
| Runtime | torch + ultralytics (~1 GB installed) | TFLite interpreter + numpy + OpenCV |

```bash
bovid predict --edge photo.jpg     # exactly what runs on a device
```

Measured end-to-end on held-out data (`scripts/compare_edge_server.py`):

| Split | Pipeline | Top-1 | Top-3 | Detector crops | ms/image | Agrees with server |
|---|---|---|---|---|---|---|
| test (n=33) | server | 48.5% | 84.8% | 93.9% | 172.4 | — |
| test | **edge** | **51.5%** | **84.8%** | 69.7% | **143.1** | 81.8% |
| valid (n=66) | server | 37.9% | 68.2% | 93.9% | 166.2 | — |
| valid | **edge** | **42.4%** | **69.7%** | 56.1% | **141.5** | 89.4% |

**The edge path loses no accuracy** — it matches or slightly exceeds the server path on both
splits (differences are within noise at these sample sizes). Top-1 agreement is 82–89%; the
disagreements are concentrated where the two detectors crop differently, and the deterministic
fallback absorbs the rest.

A useful cross-check: these figures are *identical* to the yolov8n row measured in the T13
detector comparison, which ran through Ultralytics rather than this module. Since the edge path
reimplements letterboxing, head decoding and NMS in numpy, matching Ultralytics' detection rate
(69.7% / 56.1%) and downstream accuracy exactly is strong evidence the reimplementation is
faithful rather than accidentally similar.

Shared code, not parallel code: box selection (`crop.select_animal_box`), padding, and the
deterministic fallback are the *same functions* on both paths. Only the model backends differ, so
the two deployments cannot silently drift apart.

> The ~140 ms/image figure is a MacBook number and is not the edge claim — on this machine the
> server path uses the MPS GPU while TFLite runs on CPU. T16 benchmarks a real low-power device.

### Installing: two paths, and only one of them needs torch

```bash
pip install -r requirements.txt          # dev / server: training, evaluation, demo app
pip install "bovid[edge]"                # a device: inference only, no torch
```

The package's **core dependencies are the torch-free minimum** — opencv, Pillow, numpy, PyYAML.
ultralytics and torch live in `[server]`. That split is what makes the edge claim real rather
than aspirational: measured in a clean Python 3.12 venv, `bovid[edge]` installs **12 packages,
217 MB**, with neither torch nor ultralytics present, and runs a correct prediction.

**Weights are not shipped in the wheel** (yolov8x alone is 137 MB, and weights change on a
different cadence than code), so the model root is resolved at import:

1. `BOVID_MODELS_ROOT` — a deployment storing weights anywhere;
2. `<repo>/models` — a source checkout, the usual development case;
3. `./models` — an installed `bovid` run from a directory holding the models.

```bash
BOVID_MODELS_ROOT=/opt/bovid/models bovid predict --edge photo.jpg
```

> This resolver exists because of a bug that **an editable install structurally cannot reveal**.
> With `pip install -e .` the package stays inside the repo, so deriving paths from the package
> location works by accident. A real install puts it in `site-packages`, every model path points
> somewhere that cannot exist, and the edge deployment fails on first use. It was caught only by
> installing into a clean venv and actually running the binary — worth repeating before any
> release. The same test surfaced PyYAML being undeclared and satisfied only transitively
> through ultralytics, which the edge install does not have.

### Edge benchmark (measured)

`scripts/benchmark_edge.py`, 30 images, Apple M2 CPU (TFLite never touches the GPU).

**What has to ship — 33.3 MB total:**

| Artifact | Size |
|---|---|
| Detector `yolov8n_float32.tflite` | 12.9 MB |
| Classifier `best_float32.tflite` | 20.4 MB |
| Class names + metadata YAML | <0.1 MB |
| **Total** | **33.3 MB** |

For comparison the server path needs 147 MB of weights *plus* a torch + ultralytics install of
roughly 1 GB. The edge build needs neither.

**Latency and startup**, swept across thread counts because a low-power board has far fewer cores
than a dev machine:

| Threads | Cold start | Detect | Crop | Classify | **Total/image** | p95 | Peak RSS |
|---|---|---|---|---|---|---|---|
| 1 | 0.77 s | 120.1 ms | ~0 ms | 21.6 ms | **143.7 ms** | 155.0 ms | 237 MB |
| 2 | 0.26 s | 73.8 ms | ~0 ms | 13.3 ms | **88.7 ms** | 90.1 ms | 237 MB |
| 4 | 0.21 s | 54.6 ms | ~0 ms | 10.0 ms | **65.9 ms** | 87.9 ms | 239 MB |

Readings:

- **The detector is the whole cost** — 120 ms of the 144 ms single-thread total (83%). The
  classifier is 22 ms and the crop is free. Any future optimisation belongs in the detect stage;
  shrinking the classifier further would buy almost nothing (which is a second, independent
  reason the INT8 classifier work was not worth rescuing).
- **Cold start is well under a second**, so opening the app is not a visible wait. Measured in a
  fresh subprocess, since interpreter construction and the first allocating inference happen once
  per process and an in-process timer would miss them.
- **Peak RSS ~237 MB** and flat across thread counts — comfortable on a 1 GB board, and memory
  rather than CPU is usually the binding constraint there.
- The 1-thread cold start (0.77 s) is inflated by a cold OS file cache on the first subprocess,
  not by threading; the 0.21–0.26 s figures are the representative ones.

> **This is an Apple M2 CPU, not a farmer's device.** The 1-thread column is the honest
> worst-case proxy for a single low-power core, but a Raspberry Pi 4 core is several times slower
> than an M2 core, so expect a few hundred ms to ~1 s per image there. Confirming that requires
> running this script on real hardware — the script is written to be run unchanged on the target.

#### A recurring hazard: stale artifacts

Three separate times, a stale file nearly produced a wrong conclusion — `.tflite` files dated
*1 Jan 1980*, a stale `best.onnx`, and a `saved_model/` directory that turned out to be a
**different model exported on Colab in September 2025** (`/content/sih2025/…`). The dynamic-range
INT8 measurement was initially run against that stale SavedModel and reported *better than
baseline* accuracy before the provenance check caught it. Every exported artifact is now
regenerated from the current `best.pt` in one traceable step, and metadata `date`/`description`
is worth checking before trusting any measurement.

#### Building the TFLite artifacts

TensorFlow has **no wheel for Python 3.14**, which is what the project venv runs; this is why the
previously committed `.tflite` files carried a 1980 timestamp (they were built in Colab, not
reproducibly). The export therefore runs in a dedicated Python 3.12 venv:

```bash
python3.12 -m venv .venv-export
.venv-export/bin/pip install -e . tensorflow onnx onnxruntime onnxslim onnxscript \
    onnx2tf sng4onnx onnx_graphsurgeon ai_edge_litert
.venv-export/bin/python -m bovid.cli export --model models/pretrained/yolov8n.pt \
    --formats tflite_fp32 --imgsz 640 --out-dir models/pretrained
```

Inference does **not** need this venv — only regenerating the artifacts does.

## 3. Dataset & data-quality findings

- **Source:** a Roboflow-exported dataset (`indianbovine1`) — 702 train / 66 val / 33 test images
  across 22 breeds (~32 images/breed before augmentation).
- **Augmentation-sibling structure (critical):** the 795 files are only **~330 unique source
  photos**, each exported as ~3 augmented copies (`<source>.rf.<hash>.<ext>`). This is central to
  correct evaluation (see §5).
- **Baked-in contamination:** a real fraction of images (OCR flags ~4.5%, but manual review of two
  60-image samples found ~9–17% — OCR misses low-contrast watermarks) contain overlays burned into
  the pixels: stock-photo watermarks (Getty, Shutterstock, Alamy, Fiverr), video-player UI and
  "subscribe" captions (from scraped YouTube frames — the same caption appears across *multiple*
  breeds), publication credits, and timestamps. Location analysis showed **~90% of overlays sit at
  the image edges (74% at the bottom)** — which is what makes the crop stage remove most of them.
- **Honest limitation:** ~32 images/breed is the dominant accuracy ceiling. No preprocessing or
  augmentation trick overcomes data scarcity; more labelled images is the only true ceiling-raiser
  (out of current scope).

## 4. Preprocessing & augmentation

- **Cropping (C1):** the hybrid detector-crop above. Chosen after measuring that a plain
  COCO detector reliably crops 85% of train / 94% of val-test, and that overlays are edge-located.
- **Watermark-robustness augmentation (D):** synthetic overlays (watermarks/captions/timestamps
  mimicking the real ones) are stamped onto a copy of each training image, so the model sees each
  image both clean and overlaid and learns overlays carry no breed signal — neutralising the
  residual overlays the crop can't remove and any spurious-correlation risk.
- **Real-world augmentation (step 2):** the augmented copies additionally receive random
  phone-capture corruptions that standard RandAugment does not cover — **motion blur, low-quality
  JPEG recompression, brightness/exposure shifts** — reflecting the conditions a farmer's upload
  actually faces.
- **Test-Time Augmentation (TTA):** at inference, predictions over the image and its horizontal
  flip are averaged. It is inference-only (no retraining) and measured alongside the base model.
- Class imbalance is handled by oversampling minority breeds to the majority count within each
  training split only (never the held-out data).

## 5. Evaluation methodology (why we can trust the numbers)

**The single fixed 33-image test split is not trustworthy.** With 33 images each is worth ~3
percentage points, so the same model can score anywhere in a ~10-point band by luck — which is why
early crop/augmentation changes could not be distinguished from noise.

**Primary metric: grouped 5-fold cross-validation over all 795 images.**
- **Stratified _Grouped_ K-Fold**, keyed on the source-photo prefix (before `.rf.`), so all ~3
  augmentation siblings of one photo stay in the *same* fold. This is mandatory: a naïve random
  k-fold scatters near-identical siblings across train and val and **leaks**, producing a fake
  ~93% accuracy versus the real ~50%. (Verified: 0 group overlap across folds; and the original
  Roboflow split has 0 groups spanning splits, confirming the fixed test number is itself clean.)
- The crop is applied once *outside* the loop (deterministic, no leakage); oversampling and
  augmentation happen *inside* each fold on its train portion only.
- Every image is held-out in exactly one fold → accuracy is measured on all 795 with an error bar,
  and every image also contributes to training in the other folds ("use all the data").

**Baseline result (crop + oversample + watermark-aug, stage-1):**
Top-1 **53.5% ± 4.5%**, Top-3 **78.2% ± 2.9%**. The ±4.5% std is the noise band: a change counts as
real only if it moves the CV mean beyond it. *(Real-world-aug + TTA and label-audit results pending.)*

## 6. Key engineering decisions

- **One shared crop for train and inference** — prevents train/serve skew.
- **Grouped CV as the source of truth**, not the fixed split — the leakage catch above.
- **Measured, not assumed, hyperparameters** — e.g. the uncertainty threshold and the crop's
  confidence/coverage gates were derived from measured behaviour, not guessed.
- **Root-cause over defensive error handling** — no blanket try/except; failures are surfaced loud
  at genuine boundaries and fixed at the source.
- **Pinned `ultralytics==8.3.195`** (the version that trained the deployed weights) for reproducibility.
- **Edge as a first-class path**, not an afterthought — tiny TFLite detector + INT8 classifier +
  torch-free runtime, because the project's purpose is offline farmer use.

## 7. Compute & performance on Apple Silicon (measured)

Training runs on the GPU via PyTorch MPS. Several plausible optimisations were implemented and
**measured**, and most were rejected on evidence — recorded here so they are not retried blindly:

| Optimisation | Measured result | Outcome |
|---|---|---|
| Parallel dataloader workers | Ultralytics **hard-forces `workers=0`** whenever the device is `mps`/`cpu` (`trainer.py`) | Unavailable upstream |
| Batched detector cropping | **~10× slower** *and* only 17/32 crops identical — batched inference shifts float outputs, flipping detections near the confidence gate | **Rejected** (also risked train/serve skew) |
| `device=mps` for classifier inference | **1.00× — no speedup** (~10 ms/img either way); the classifier is small enough that MPS launch overhead cancels the gain | Kept for correctness, not speed |
| torch thread count (8 / 4 / 2) | 160 / 157 / 159 ms/img — **within noise** | No effect |
| Mixed precision (AMP) | Already enabled by default | — |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | Prevents a hard abort when an op has no MPS kernel | **Adopted** (robustness) |

**The binding constraint is 8 GB of unified memory shared between CPU and GPU**, not core count —
which is also why running several folds in parallel is counterproductive on this machine.

A useful consequence for the edge goal: the classifier already runs at **~10 ms/image on CPU**,
so the deployment target does not need a GPU at all.

## 8. Honest limitations

- Small dataset (~32 images/breed) caps achievable accuracy.
- Visually similar zebu breeds confuse (e.g. Gir↔Banni, Kankrej↔Kangayam↔Alambadi).
- Dataset contamination (watermarks/captions) is mitigated, not eliminated.
- The classifier is assistive; it should not be treated as an authoritative breed determination.
