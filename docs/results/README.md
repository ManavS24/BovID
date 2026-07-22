# Measured results

The evidence behind the numbers in [../METHODOLOGY.md](../METHODOLOGY.md). Each file is regenerable
by the script named below; they are archived here because they are the receipts for the project's
"measured, not assumed" claims.

| File | What it measures | Regenerate with |
|---|---|---|
| `cv_results_step2.json` | Grouped 5-fold CV, real-world-aug + TTA ablation | `scripts/kfold.py` |
| `cv_results_step3.json` | Grouped 5-fold CV, label-audit ablation | `scripts/kfold.py` |
| `detector_comparison_edge_test.json` | yolov8x vs yolov8n(.pt/.tflite) breed accuracy, test split | `scripts/compare_detectors.py` |
| `detector_comparison_edge_valid.json` | same, valid split | `scripts/compare_detectors.py` |
| `classifier_comparison.json` | FP32 vs INT8 classifier accuracy (the INT8 rejection) | `scripts/compare_classifiers.py` |
| `edge_vs_server.json` | torch-free edge pipeline vs torch server, held-out | `scripts/compare_edge_server.py` |
| `edge_benchmark.json` | edge size / cold start / latency / RSS | `scripts/benchmark_edge.py` |
