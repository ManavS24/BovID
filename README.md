# bovid — Indian Bovine Breed Classifier

Identify the breed of an Indian cow or buffalo from a single photograph. A two-stage
**detect → crop → classify** pipeline over **22 indigenous breeds**, built to run **offline on a
farmer's phone or low-power device** — no server, no connectivity, no PyTorch on the device.

**▶ [Live demo](REPLACE_WITH_YOUR_DEPLOY_URL)** — the torch-free edge pipeline running in the
browser. Upload a cow/buffalo photo or click an example. **[▶ Watch a 60s demo](docs/demo.mp4).**

<!-- Inline player on GitHub: after pushing, either drag docs/demo.mp4 into the README via
GitHub's web editor (auto-embeds a player), or set the raw URL in the tag below:
<video src="https://raw.githubusercontent.com/<user>/BovID/main/docs/demo.mp4" controls muted></video> -->

```bash
bovid predict cow.jpg
#  - Red_Sindhi: 0.783
#  - Gir: 0.126
#  - Vechur: 0.090
#    note: top-1 below 90% — treat as uncertain.
```

> **Honest framing up front.** This is an *assistive* classifier on a small, real-world dataset
> (~32 images/breed). It is genuinely useful as a first guess with confidence, and it is honest
> about when it is unsure. It is **not** an authoritative breed determination, and the results
> below are the real cross-validated numbers, not a flattering single split. See
> [Results](#results-measured-honestly) and [Limitations](#limitations).

---

## What it does

Given one photo, the pipeline:

1. **Detects** the animal with a COCO-pretrained detector and **crops** tightly to it — removing
   the baked-in captions/watermarks that pollute web-scraped cattle images. If no animal is found
   confidently, it falls back to a deterministic bottom/edge trim (where ~90% of those overlays
   live) rather than failing.
2. **Classifies** the crop into one of 22 breeds with a YOLOv8s classifier, returning the top-3
   with confidence and an explicit **uncertain** flag when the top guess is below a *measured*
   threshold.

The crop is **shared code between training and inference**, so the model always sees the same
distribution it was trained on — no train/serve skew.

```
raw image ──▶ [1] detector ──▶ crop to animal ──▶ [2] breed classifier ──▶ top-3 + confidence
                   │  (fallback if no confident detection)      │
                   └──▶ deterministic bottom/edge-trim + center ┘
```

## Results (measured honestly)

The headline numbers come from **grouped 5-fold cross-validation over all 795 images**, not a
single test split:

| Metric | Score |
|---|---|
| **Top-1 accuracy** | **53.5% ± 4.5%** |
| **Top-3 accuracy** | **78.2% ± 2.9%** |

Two things make these trustworthy:

- **The split can't leak.** The dataset is Roboflow-augmented: each source photo has ~3
  near-identical copies (795 files, only 328 unique source photos). A naive random split scatters
  those siblings across train and validation and reports a **fake ~93%**. Grouped K-Fold keyed on
  the source photo keeps all siblings in the same fold — the real number is ~53%. This leakage
  catch is the single most important methodological decision in the project.
- **Every image is measured.** Each is held out in exactly one fold, so the accuracy is over all
  795 images with a real error bar (the ±4.5%), not over 33 lucky ones.

Chance across 22 breeds is 4.5%, so top-1 is ~12× chance and top-3 clears 78%. The ceiling is
**data scarcity** (~32 images/breed) and **genuine visual similarity** between zebu breeds
(Gir↔Banni, Kankrej↔Kangayam), not a fixable modelling bug — established by measurement, not
assumed.

## Runs offline on a device

The edge deployment is a first-class path, not an afterthought. It runs the same pipeline with
**no PyTorch and no Ultralytics** — a TFLite detector + classifier behind a torch-free runtime:

| | Server (`bovid predict`) | Edge (`bovid predict --edge`) |
|---|---|---|
| Detector | yolov8x — 137 MB, torch | yolov8n TFLite — 13 MB |
| Classifier | best.pt — 10 MB, torch | FP32 TFLite — 20 MB |
| Runtime | torch + ultralytics (~1 GB) | TFLite interpreter + numpy + OpenCV |
| **Install** | `pip install "bovid[server]"` | `pip install "bovid[edge]"` — **12 packages, 217 MB** |

Measured on an M2 CPU (30 images):

- **33.3 MB** total to ship · **cold start 0.21 s** · **~144 ms/image** single-threaded (66 ms at
  4 threads) · peak RSS **237 MB**.
- **No accuracy lost** versus the server path — the edge pipeline scores 51.5% / 42.4% top-1 on
  the two held-out splits, matching or slightly beating the torch server, with 82–89% top-1
  agreement.

> Every one of these numbers is reproducible: `python scripts/benchmark_edge.py` and
> `python scripts/compare_edge_server.py`. The M2 CPU is a *proxy* for a device, not a device —
> a Raspberry Pi core is slower, so expect a few hundred ms to ~1 s there.

Two design calls worth calling out, both made on evidence and recorded in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md):

- **INT8 quantisation was measured and rejected** — every variant collapsed to ~chance (the model
  mode-collapses to 4 of 22 classes). The edge classifier ships FP32, which is *bit-faithful* to
  the torch model (100% agreement).
- **The tiny detector's low detection rate is a non-issue** — a missed detection just takes the
  deterministic fallback crop, so it costs no measurable breed accuracy.

## Quickstart

Requires Python ≥ 3.10. The raw dataset and the project's trained weights are in the repo; the
stock Ultralytics detector bases are fetched on setup (they're large and freely downloadable).

```bash
# Server / development install (training, evaluation, demo app)
pip install -e ".[server,app,dev]"

# Fetch the stock detector/classifier bases (server path only; the edge path needs none)
python scripts/fetch_weights.py

# Predict on an image
bovid predict path/to/cow.jpg

# Launch the demo (upload a photo or click an example)
streamlit run app.py

# Evaluate on the held-out test split
bovid evaluate
```

Edge install — no torch, for a device:

```bash
pip install "bovid[edge]"
bovid predict --edge path/to/cow.jpg
# weights aren't shipped in the wheel; point at them if not running from a checkout:
BOVID_MODELS_ROOT=/path/to/models bovid predict --edge cow.jpg
```

## Project structure

```
bovid/                 the package (importable, torch-free core)
  config.py            single source of truth; every knob overridable via BOVID_* env vars
  crop.py              shared detect+crop stage (training and inference use the same code)
  predict.py           server inference API (PredictionResult contract)
  edge.py              torch-free edge pipeline — same contract, TFLite backends
  tflite_detector.py   numpy YOLOv8 decode + NMS (no torch)
  tflite_backend.py    TFLite classifier runner
  train.py evaluate.py export.py dataset.py augment.py audit.py   pipeline stages
  cli.py               `bovid` entry point (predict / evaluate / train / export / audit)
app.py                 Streamlit demo
scripts/               CV, detector/classifier/edge comparisons, benchmark, dataset audit
tests/                 unit + integration + end-to-end
docs/METHODOLOGY.md    the full design, data-quality findings, and every measured decision
```

## Testing

```bash
pytest -m "not slow"     # unit + integration + edge parity (the CI gate)
pytest                   # everything, incl. a tiny real training run and subprocess e2e
```

The suite has two entry points: `tests/unit_tests.py` (isolated, runs anywhere) and
`tests/integration_tests.py` (real models/artifacts, skips cleanly when absent). CI (GitHub
Actions) runs ruff + mypy + the tests on every push. Tests that need the 137 MB server detector
**skip cleanly** when it is absent, so CI stays green on a bare checkout. The CI gate
(`-m "not slow"`) is **111 passing** on Python 3.12 + TensorFlow, and 110 passing / 1 skipped on
the Python 3.14 dev venv (where TensorFlow has no wheel, so the FP32-parity test skips).

## Limitations

- **Small dataset** (~32 images/breed) is the main accuracy ceiling.
- **Visually similar zebu breeds** are genuinely hard to separate.
- **Dataset contamination** (watermarks/captions) is mitigated by the crop stage, not eliminated.
- **Assistive, not authoritative** — always surfaced with confidence and an uncertainty flag.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — exact install steps for the server, edge, and export
  profiles (which Python, which extras, which need TensorFlow).
- [docs/USAGE.md](docs/USAGE.md) — CLI, demo app, Python API, and edge/server deployment.
- [docs/DEMO.md](docs/DEMO.md) — a rehearsed ~5-minute demo runbook.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow, module responsibilities, and the
  server-vs-edge paths, with diagrams.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — model card (intended use, metrics, limitations,
  ethics) + a public API reference.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — data-quality findings, evaluation methodology, and
  every engineering decision (including the ones measured and *rejected*).
