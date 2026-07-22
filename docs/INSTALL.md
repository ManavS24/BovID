# Installation & reproducible setup

Three install profiles, each verified in a clean virtual environment. Pick the one matching what
you want to do — **you do not need TensorFlow or PyTorch unless the profile says so.**

| Profile | For | Python | Heavy deps |
|---|---|---|---|
| **Server / dev** | training, evaluation, the demo app, running the test suite | 3.10–3.14 | PyTorch (via ultralytics) |
| **Edge** | on-device inference only | 3.10–3.14 | none — no torch, no TensorFlow |
| **Export** | regenerating the `.tflite` / ONNX artifacts | **3.12** | TensorFlow |

Always install into a **virtual environment**, never the system Python.

---

## 1. Server / development

Everything except artifact export. Works on any supported Python, including 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[server,app,dev]"
```

Verify:

```bash
bovid predict datasets/indianbovine1/test/*.jpg | head    # a real prediction
pytest -m "not slow"                                       # the test suite (CI gate)
streamlit run app.py                                       # the demo, at http://localhost:8501
```

`[server]` pulls ultralytics (and PyTorch); `[app]` adds Streamlit; `[dev]` adds pytest + ruff.

## 2. Edge (on-device) — no PyTorch, no TensorFlow

The offline deployment profile. Uses `ai-edge-litert` (a standalone TFLite runtime with wheels on
every supported Python, including 3.14) — **not** TensorFlow.

```bash
python3 -m venv .venv-edge
source .venv-edge/bin/activate
pip install "bovid[edge]"
```

This installs ~12 packages / ~220 MB, with neither torch nor ultralytics present. Verify:

```bash
python -c "import sys, bovid.edge; print('torch loaded:', 'torch' in sys.modules)"   # -> False
bovid predict --edge path/to/cow.jpg
```

**Weights are not shipped inside the package** (the models are large and versioned separately),
so an installed `bovid` needs to be told where they are:

```bash
BOVID_MODELS_ROOT=/path/to/models bovid predict --edge cow.jpg
```

If you run from a source checkout (with `models/` present), no env var is needed — the repo
layout is found automatically. The error message names this fix if weights are missing.

## 3. Export (regenerating deployment artifacts)

Only needed to rebuild the `.tflite`/ONNX files from `best.pt`. This is the **one** profile that
requires a specific Python: the `onnx2tf` → TensorFlow toolchain has **no wheel for Python 3.13 or
3.14**, so use **3.12**.

```bash
python3.12 -m venv .venv-export
source .venv-export/bin/activate
pip install -e ".[server,export]" tensorflow
bovid export --model models/best/best.pt --formats tflite_fp32 --imgsz 224 --out-dir models/best
```

> Why a separate venv: keeping TensorFlow out of the main environment is deliberate — it lags
> Python releases and would otherwise pin the whole project to an older Python. The dev venv runs
> 3.14; only export needs the 3.12 + TF venv. See [METHODOLOGY.md](METHODOLOGY.md) §2.

## Fresh clone: weights & data

What is committed vs. obtained on setup:

- **Tracked in the repo:** the raw dataset (`datasets/indianbovine1/`), the project's own trained
  weights (`models/best/best.pt`, `best_float32.tflite`) and the edge TFLite detector
  (`models/pretrained/yolov8n_float32.tflite` + names/metadata). The **edge path works from a
  clean clone with no extra steps.**
- **Fetched on setup** — the stock Ultralytics bases (`yolov8x.pt` is 131 MB, over GitHub's limit;
  all three are freely downloadable), needed only for the **server** path and for training:

  ```bash
  python scripts/fetch_weights.py     # -> models/pretrained/{yolov8x,yolov8s-cls,yolov8n}.pt
  ```

- **Regenerated, not committed** — the cropped datasets (`datasets/bovine_crops`,
  `datasets/bovine_cls`) are derived from `indianbovine1` and rebuilt by the training pipeline
  (`bovid train`). They are needed only to **retrain or run cross-validation**, never for
  inference.

`pyproject.toml` is the single source of truth for dependencies; `requirements.txt` installs the
full dev setup, and `requirements-lock.txt` pins exact versions.

### Environment overrides

Any configuration constant is overridable via a `BOVID_<NAME>` environment variable — no code
edits. The ones that matter for setup:

| Variable | Effect |
|---|---|
| `BOVID_MODELS_ROOT` | Where to find `models/` (weights not shipped in the wheel). |
| `BOVID_DATA_ROOT` | Where to find `datasets/`. |
| `BOVID_DETECTOR_MODEL` | Swap the crop detector (e.g. point the server path at the tiny TFLite one). |
| `BOVID_DEVICE` | Force `cpu` / `mps` / `cuda:0` instead of auto-detecting. |

## Reproducibility notes

- **Pinned `ultralytics==8.3.195`** — the version the deployed weights were trained and exported
  with. Do not bump it without re-validating.
- The test suite is the reproducibility check: `pytest -m "not slow"` should report all-pass with
  the TFLite tests either running (if a runtime is installed) or skipping cleanly. On a checkout
  without the 137 MB `yolov8x.pt`, the detector-dependent tests skip rather than fail.
- A clean-room verification (fresh clone → install → tests → predict → app) is the final release
  gate.
