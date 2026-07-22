# Usage & deployment

How to run `bovid` — as a command-line tool, as the demo app, as a Python API, and on an edge
device. For installing the right profile first, see [INSTALL.md](INSTALL.md).

## Command line

The `bovid` command is installed with the package (`pip install -e .`). Diagnostics go to
**stderr**, results to **stdout**, so predictions can be piped cleanly.

```bash
bovid predict cow.jpg                 # one image
bovid predict a.jpg b.jpg c.jpg       # several, in order
bovid predict --edge cow.jpg          # the torch-free device path (see Edge, below)
bovid predict --topk 5 cow.jpg        # more candidates
bovid predict --model path/to/other.pt cow.jpg
bovid predict cow.jpg > result.txt    # clean: only results on stdout
```

Output:

```
Top-3 predictions for cow.jpg  [crop: detector]
 - Red_Sindhi: 0.783
 - Gir: 0.126
 - Vechur: 0.090
```

When the top-1 confidence is below the measured threshold, a `note: … treat as uncertain` line is
added — the model is telling you it is guessing. `[crop: detector]` vs `[crop: fallback]` reports
which crop path fired (a confident animal detection, or the deterministic margin trim).

Other subcommands:

```bash
bovid evaluate                        # top-1/top-3 + per-class report on the held-out test split
bovid export --formats tflite_fp32    # regenerate deployment artifacts (needs the export profile)
bovid audit                           # OCR-scan the dataset for baked-in watermarks
bovid -v predict cow.jpg              # -v anywhere for debug-level logging
```

## Demo app

```bash
streamlit run app.py                  # opens http://localhost:8501
```

Upload a photo or click one of the built-in examples. The app shows the **input** beside the
**exact crop the classifier sees** (labelled with which crop path produced it), the top-3 with
confidence bars, a confident/uncertain verdict, and — in the sidebar — the honest measured
accuracy and known limitations. It is the fastest way to demonstrate the whole pipeline.

## Python API

```python
from bovid.predict import load_model, predict

model = load_model()                          # loads models/best/best.pt once
result = predict("cow.jpg", model=model)

result.top.breed                              # 'Red_Sindhi'
result.top.confidence                         # 0.783
result.is_confident                           # False (below the measured threshold)
result.crop_method                            # 'detector' or 'fallback'
for breed, conf in result.predictions:        # top-k, highest first
    print(breed, conf)
```

The torch-free edge pipeline returns the **identical** `PredictionResult`:

```python
from bovid.edge import EdgePipeline

pipeline = EdgePipeline()                      # load once; interpreter build is the costly part
result = pipeline.predict("cow.jpg")           # same object as predict() above
```

Load the model / pipeline **once** and reuse it — construction (weights load, interpreter
allocation) dominates a single call, especially on a low-power device.

## Edge deployment (phone / Raspberry Pi / low-power)

The edge path runs the full `detect → crop → classify` pipeline with **no PyTorch and no
TensorFlow** — just a TFLite interpreter, NumPy and OpenCV. It is the whole reason the project
exists: offline breed ID where there is no connectivity.

**1. Install the edge profile** (any Python 3.10–3.14; see [INSTALL.md](INSTALL.md#2-edge-on-device--no-pytorch-no-tensorflow)):

```bash
pip install "bovid[edge]"
```

**2. Make the weights reachable.** They are not shipped inside the wheel. Copy `models/` to the
device (the edge artifacts total ~33 MB — the two TFLite models) and point `bovid` at them:

```bash
BOVID_MODELS_ROOT=/opt/bovid/models bovid predict --edge cow.jpg
```

**3. Tune threads for the hardware.** The `EdgePipeline` accepts `num_threads`; a small board
with few cores does best at 1–2. From Python:

```python
from bovid.edge import EdgePipeline
pipeline = EdgePipeline(num_threads=2)
```

**What to expect (measured on an M2 CPU as a proxy — a Pi core is slower):** ~33 MB to ship,
sub-second cold start, ~144 ms/image single-threaded, ~237 MB peak RAM, and **no accuracy loss**
versus the server path. Reproduce on the target with:

```bash
python scripts/benchmark_edge.py --threads 1 2 4      # size, cold start, per-stage latency, RSS
python scripts/compare_edge_server.py                 # edge vs server accuracy + agreement
```

The edge classifier ships **FP32**, not INT8 — INT8 was measured and rejected because it collapsed
to near-chance accuracy on this model (see [METHODOLOGY.md](METHODOLOGY.md) §2).

## Server deployment

Run the Streamlit app directly, behind a process manager:

```bash
pip install -e ".[server,app]"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

Two images ship with the repo:

```bash
# Server: the full torch pipeline + Streamlit demo
docker build -t bovid .
docker run -p 8501:8501 bovid          # open http://localhost:8501

# Edge: the torch-free TFLite pipeline, the way a device runs it (~33 MB of models, no torch)
docker build -f Dockerfile.edge -t bovid-edge .
docker run --rm -v "$PWD/photo.jpg:/app/photo.jpg" bovid-edge predict --edge photo.jpg
```

The server image bakes in the weights and the test-split example images, so `docker run` is a
working demo with nothing to mount. The edge image carries only the two tiny TFLite models; mount
the photo you want classified.

## Configuration

Every setting is overridable via a `BOVID_<NAME>` environment variable — no code edits. Common
ones:

| Variable | Default | Use |
|---|---|---|
| `BOVID_MODELS_ROOT` | repo `models/` | where weights live (required for an installed, non-checkout run) |
| `BOVID_DETECTOR_MODEL` | `yolov8x.pt` | swap the crop detector (e.g. the tiny TFLite one on the server path) |
| `BOVID_DEVICE` | auto | force `cpu` / `mps` / `cuda:0` |
| `BOVID_CONFIDENCE_THRESHOLD` | `0.9` | the uncertain-flag cutoff |
| `BOVID_DEFAULT_TOPK` | `3` | how many candidates to return |

Example — run the server pipeline but with the tiny detector, on CPU:

```bash
BOVID_DETECTOR_MODEL=models/pretrained/yolov8n_float32.tflite BOVID_DEVICE=cpu bovid predict cow.jpg
```
