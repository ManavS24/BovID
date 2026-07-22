# Demo runbook

A rehearsed ~5-minute walkthrough. Every command and number below was run live and is current.
Setup once: `pip install -e ".[server,app,dev]"` and `python scripts/fetch_weights.py`.

> One honest sentence to open with: *"It's a first guess with a confidence, not an authority —
> and it tells you when it's unsure."* Say the accuracy numbers plainly; they're modest and
> real, and the methodology is the strong part.

---

## Beat 1 — the app (the visual hook)

```bash
streamlit run app.py        # http://localhost:8501
```

Click an example, or upload a cow/buffalo photo. Point out, in order:
1. **Input photo vs. "what the classifier sees"** — the crop, labelled `detector` or `fallback`.
   This makes the detect→crop→classify pipeline *visible*, not a black box.
2. **Top-3 with confidence bars.**
3. **The confident/uncertain banner** — and that the 90% threshold was *measured, not guessed*.
4. **Sidebar:** the honest CV accuracy and the known limitations, in the product itself.

## Beat 2 — the CLI, and the two-model story

```bash
IMG=datasets/indianbovine1/test/00000133_jpg.rf.c7116f67a784ebed9fe37772fe2b51e6.jpg
bovid predict "$IMG"              # server: torch + yolov8x
bovid predict --edge "$IMG"       # edge: torch-free TFLite
```

Both return **Red Sindhi** top-1 (server 0.78, edge 0.69), same pipeline, same result contract —
the only difference is the backend. Segue: *"the edge one runs with no PyTorch at all."*

## Beat 3 — the edge story (the differentiator)

The whole point: **offline breed ID on a farmer's device.** Numbers (measured, M2 CPU):

| | Server | Edge |
|---|---|---|
| Ships | ~147 MB weights + ~1 GB torch | **33.3 MB total, no torch** |
| Install | `bovid[server]` | `bovid[edge]` — 12 packages |
| Latency (1 core) | — | **~144 ms/image**, 66 ms at 4 cores |
| Cold start / RAM | — | **~0.2 s / 237 MB** |
| Accuracy | baseline | **no loss** (matches server) |

```bash
python scripts/benchmark_edge.py --threads 1 2 4      # reproduce the table live if asked
docker run --rm -v "$PWD/photo.jpg:/app/photo.jpg" bovid-edge predict --edge photo.jpg
```

Two decisions to mention as "measured, then rejected": **INT8** quantisation (collapsed to
chance — shipped FP32 instead) and the worry that the **tiny detector** would hurt accuracy (it
doesn't — a missed detection just uses the fallback crop).

## Beat 4 — results, honestly

```bash
bovid evaluate                    # held-out test split: Top-1 48.5%, Top-3 84.8%
```

The number that matters, and *why it's trustworthy*:

- **Grouped 5-fold CV over all 795 images: Top-1 53.5% ± 4.5%, Top-3 78.2% ± 2.9%.**
- The dataset is Roboflow-augmented — 795 files, only ~328 unique photos. A naive split scatters
  a photo's copies across train/val and **leaks, reporting a fake ~93%.** Grouping by source
  photo keeps them together; the real number is ~53%. **This leakage catch is the strongest part
  of the project** — lead with it.
- Chance is 4.5% (22 breeds), so top-1 is ~12× chance and top-3 clears 78%.

## Beat 5 — limitations (say these first, not if asked)

- **~32 images/breed** — data scarcity is the ceiling, not the model.
- **Zebu breeds genuinely look alike** (Gir/Banni, Kankrej/Kangayam) — that's why top-3 ≫ top-1.
- **Assistive, not authoritative** — surfaced with confidence and an uncertainty flag.
- Dataset watermarks are *mitigated* by the crop, not eliminated.

## If something breaks

- App won't start → check `streamlit run app.py` logs; port 8501 free?
- `bovid predict` errors on weights → `python scripts/fetch_weights.py` (server detector isn't
  committed).
- No GPU / wrong device → `BOVID_DEVICE=cpu bovid predict …`.
- Fall back to the CLI (Beat 2) — it needs nothing but the weights and shows the same result.

## One-paragraph summary (if you have 30 seconds)

> A two-stage detect→crop→classify pipeline identifies 22 Indian bovine breeds from one photo.
> It runs **offline on a device** in a 33 MB, torch-free build with no accuracy loss versus the
> server. Accuracy is **53.5% top-1 / 78.2% top-3**, measured with grouped cross-validation that
> avoids the data-leakage trap a naive split falls into (which would have reported a misleading
> ~93%). Every design choice — the crop strategy, the edge detector, dropping INT8 — was decided
> by measurement, and it's honest about being an assistive tool on a small dataset.
