"""Integration suite (one of the two entry points): real modules wired against real artifacts
(detector, classifier, TFLite exports, dataset, CLI, app). Each declares its resources via the
_bovid_reqs.py guards and skips cleanly when they are absent."""

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from _bovid_reqs import (
    requires_dataset,
    requires_deployed_model,
    requires_detector,
    requires_tensorflow,
    requires_tflite,
    valid_breeds,
)

from bovid import config

FP32_CLASSIFIER = os.path.join(config.BEST_MODEL_DIR, "best_float32.tflite")
EDGE_DETECTOR = config.EDGE_DETECTOR_MODEL
EDGE_CLASSIFIER = FP32_CLASSIFIER

_has_fp32 = pytest.mark.skipif(not os.path.exists(FP32_CLASSIFIER),
                               reason=f"{FP32_CLASSIFIER} not built")
_has_edge_artifacts = pytest.mark.skipif(
    not (os.path.exists(EDGE_DETECTOR) and os.path.exists(EDGE_CLASSIFIER)),
    reason="edge artifacts not built",
)


def _test_images(n):
    test_dir = os.path.join(config.RAW_DATA_DIR, "test")
    names = sorted(p for p in os.listdir(test_dir) if p.endswith(".jpg"))[:n]
    return [os.path.join(test_dir, name) for name in names]


@pytest.fixture(scope="class")
def sample_crops():
    from bovid.crop import crop_animal, get_detector
    from bovid.utils import load_image_bgr
    detector = get_detector()
    return [crop_animal(load_image_bgr(img), detector)[0] for img in _test_images(5)]


@pytest.fixture(scope="class")
def sample_image():
    return _test_images(1)[0]


@requires_detector
@requires_deployed_model
@requires_dataset
class TestServerPredict:
    def test_predict_is_a_single_coherent_flow_with_a_valid_ranked_topk(self):
        from bovid.predict import load_model, predict

        result = predict(_test_images(1)[0], model=load_model(), topk=3)
        breeds = valid_breeds()

        assert result.crop_method in {"detector", "fallback"}
        assert result.crop_bgr is not None and result.crop_bgr.ndim == 3
        assert len(result.predictions) == 3
        for breed, conf in result.predictions:
            assert breed in breeds
            assert 0.0 <= conf <= 1.0
        confidences = [c for _, c in result.predictions]
        assert confidences == sorted(confidences, reverse=True)
        assert result.top is result.predictions[0]
        assert result.top.confidence == max(c for _, c in result.predictions)
        assert isinstance(result.is_confident, bool)


@requires_tflite
@requires_detector
@requires_dataset
@_has_fp32
class TestTFLiteRunner:
    @requires_tensorflow
    def test_fp32_runner_reproduces_torch_probabilities(self, sample_crops):
        """Asserted on probabilities, not just argmax: a near-miss preprocessing bug can leave the
        top-1 unchanged while shifting confidences."""
        from bovid.predict import load_model
        from bovid.tflite_backend import TFLiteClassifier
        from bovid.utils import bgr_to_pil

        runner = TFLiteClassifier(FP32_CLASSIFIER)
        reference = load_model(FP32_CLASSIFIER)          # ultralytics can load FP32

        for crop_bgr in sample_crops:
            mine = runner.predict(crop_bgr)
            r = reference.predict(source=bgr_to_pil(crop_bgr), imgsz=config.DEFAULT_IMGSZ,
                                  verbose=False)[0]
            theirs = (r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu")
                      else np.array(r.probs.data))
            assert mine.argmax() == theirs.argmax()
            np.testing.assert_allclose(mine, theirs, atol=1e-4)

    def test_top_k_is_ordered_and_named(self, sample_crops):
        from bovid.tflite_backend import TFLiteClassifier
        runner = TFLiteClassifier(FP32_CLASSIFIER)
        predictions = runner.top_k(sample_crops[0], k=3)
        assert len(predictions) == 3
        confidences = [c for _, c in predictions]
        assert confidences == sorted(confidences, reverse=True)
        assert all(breed in runner.names.values() for breed, _ in predictions)

    def test_probabilities_form_a_distribution(self, sample_crops):
        from bovid.tflite_backend import TFLiteClassifier
        runner = TFLiteClassifier(FP32_CLASSIFIER)
        probs = runner.predict(sample_crops[0])
        assert probs.shape == (len(runner.names),)
        assert probs.min() >= 0.0
        assert probs.sum() == pytest.approx(1.0, abs=1e-3)


@requires_tflite
@requires_dataset
@_has_edge_artifacts
class TestEdgePipeline:
    def test_edge_pipeline_imports_no_torch_or_ultralytics(self):
        """Runs in a subprocess: this process already imported torch via other tests, so an
        in-process sys.modules check would pass regardless of what bovid.edge pulls in."""
        code = ("import sys; import bovid.edge; "
                "print('torch' in sys.modules, 'ultralytics' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=config.PROJECT_ROOT)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False False", (
            f"bovid.edge pulled in a heavy dependency: torch/ultralytics = {out.stdout.strip()}"
        )

    def test_edge_returns_the_same_contract_as_the_server_path(self, sample_image):
        from bovid.edge import EdgePipeline
        from bovid.result import Prediction, PredictionResult

        result = EdgePipeline().predict(sample_image)
        assert isinstance(result, PredictionResult)
        assert result.image_path == sample_image
        assert result.crop_method in {"detector", "fallback"}
        assert result.crop_bgr is not None
        assert len(result.predictions) == config.DEFAULT_TOPK
        assert all(isinstance(p, Prediction) for p in result.predictions)
        confidences = [c for _, c in result.predictions]
        assert confidences == sorted(confidences, reverse=True)
        assert result.top is result.predictions[0]
        assert isinstance(result.is_confident, bool)

    @requires_detector
    @requires_deployed_model
    @_has_fp32
    def test_edge_pipeline_agrees_with_server_on_top1(self):
        """Asserts high agreement rather than identity: the two paths use different detectors and
        may crop differently."""
        from bovid.edge import EdgePipeline
        from bovid.predict import load_model, predict

        images = _test_images(10)
        server_model = load_model()
        edge = EdgePipeline()
        agree = sum(predict(img, model=server_model).top.breed == edge.predict(img).top.breed
                    for img in images)
        assert agree >= len(images) - 2, f"edge/server top-1 agreement too low: {agree}/{len(images)}"


@pytest.mark.slow
@requires_detector
@requires_deployed_model
@requires_dataset
class TestEndToEnd:
    @staticmethod
    def _an_image():
        return _test_images(1)[0]

    @staticmethod
    def _tflite_available():
        from importlib.util import find_spec
        return find_spec("tensorflow") is not None or find_spec("ai_edge_litert") is not None

    def test_cli_predict_prints_a_valid_breed_and_confidence(self):
        out = subprocess.run([sys.executable, "-m", "bovid.cli", "predict", self._an_image()],
                             capture_output=True, text=True, cwd=config.PROJECT_ROOT)
        assert out.returncode == 0, out.stderr
        pairs = re.findall(r"-\s+(\w+):\s+([0-9.]+)", out.stdout)     # " - Red_Sindhi: 0.783"
        assert pairs, f"no predictions parsed from stdout:\n{out.stdout}"
        breeds = valid_breeds()
        for breed, conf in pairs:
            assert breed in breeds, f"{breed!r} is not a known breed"
            assert 0.0 <= float(conf) <= 1.0

    def test_cli_results_go_to_stdout_not_stderr(self):
        """Diagnostics on stderr, results on stdout, so `predict > out.txt` stays clean."""
        out = subprocess.run([sys.executable, "-m", "bovid.cli", "predict", self._an_image()],
                             capture_output=True, text=True, cwd=config.PROJECT_ROOT)
        assert "predictions for" in out.stdout.lower()
        assert "predictions for" not in out.stderr.lower()

    def test_cli_edge_predict_matches_contract(self):
        if not self._tflite_available():
            pytest.skip("no TFLite runtime for the --edge path")
        out = subprocess.run(
            [sys.executable, "-m", "bovid.cli", "predict", "--edge", self._an_image()],
            capture_output=True, text=True, cwd=config.PROJECT_ROOT)
        assert out.returncode == 0, out.stderr
        pairs = re.findall(r"-\s+(\w+):\s+([0-9.]+)", out.stdout)
        assert pairs, f"no predictions from --edge:\n{out.stdout}"
        assert all(breed in valid_breeds() for breed, _ in pairs)

    def test_streamlit_app_renders_a_prediction_from_an_example(self):
        AppTest = pytest.importorskip("streamlit.testing.v1",
                                      reason="streamlit not installed").AppTest
        app = AppTest.from_file(os.path.join(config.PROJECT_ROOT, "app.py"), default_timeout=120)
        app.run()
        assert not app.exception, f"app raised on initial load: {app.exception}"
        assert app.button, "no example buttons rendered"
        app.button[0].click().run()
        assert not app.exception, f"app raised after clicking an example: {app.exception}"
        rendered = " ".join(m.value for m in app.markdown)   # "**<Breed>** — <pct>%"
        assert re.search(r"\*\*[\w ]+\*\*\s*—\s*[0-9.]+%", rendered), (
            f"no 'Breed — NN%' prediction rendered; markdown was:\n{rendered[:500]}"
        )


# Size ceilings leave headroom over measured (detector 12.9, classifier 20.4 MB); latency ceiling
# is ~10x the measured ~144 ms/image so it trips a real regression without flaking on slow CI.
MAX_DETECTOR_MB = 20.0
MAX_CLASSIFIER_MB = 30.0
MAX_TOTAL_SHIP_MB = 45.0
MAX_MS_PER_IMAGE = 1500.0


def _mb(path):
    return os.path.getsize(path) / 1e6


@pytest.mark.slow
@_has_edge_artifacts
class TestPerformance:
    def test_edge_detector_is_the_tiny_one(self):
        assert _mb(EDGE_DETECTOR) <= MAX_DETECTOR_MB, (
            f"edge detector is {_mb(EDGE_DETECTOR):.1f} MB (limit {MAX_DETECTOR_MB}) -- "
            "is the server yolov8x being shipped by mistake?"
        )

    def test_edge_classifier_size_within_budget(self):
        assert _mb(EDGE_CLASSIFIER) <= MAX_CLASSIFIER_MB, (
            f"edge classifier is {_mb(EDGE_CLASSIFIER):.1f} MB (limit {MAX_CLASSIFIER_MB})"
        )

    def test_total_shipped_size_within_budget(self):
        total = _mb(EDGE_DETECTOR) + _mb(EDGE_CLASSIFIER)
        assert total <= MAX_TOTAL_SHIP_MB, f"edge payload is {total:.1f} MB (limit {MAX_TOTAL_SHIP_MB})"

    @requires_tflite
    def test_edge_latency_has_not_regressed_by_an_order_of_magnitude(self):
        import statistics
        import time

        from bovid.edge import EdgePipeline

        test_dir = os.path.join(config.RAW_DATA_DIR, "test")
        images = [os.path.join(test_dir, p)
                  for p in sorted(os.listdir(test_dir)) if p.endswith(".jpg")][:8]
        if len(images) < 3:
            pytest.skip("need a few test images to time")

        pipeline = EdgePipeline()
        pipeline.predict(images[0])                       # warm up (exclude one-off allocation)
        latencies = []
        for path in images:
            t0 = time.perf_counter()
            pipeline.predict(path)
            latencies.append((time.perf_counter() - t0) * 1000)

        median = statistics.median(latencies)
        assert median <= MAX_MS_PER_IMAGE, (
            f"edge median latency {median:.0f} ms/image exceeds {MAX_MS_PER_IMAGE} ms -- a ~10x "
            "regression; check for a double detector run or a delegate fallback"
        )


CROPS_TRAIN = os.path.join(config.YOLO_DATA_DIR, "train")
TRAIN_N, VAL_N = 6, 2   # per-breed fixture counts; everything else is derived from these


@pytest.mark.slow
class TestTraining:
    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """A 2-class train/val layout built from real crops (so the model can actually fit it)."""
        import shutil
        if not os.path.isdir(CROPS_TRAIN):
            pytest.skip(f"no crops at {CROPS_TRAIN} to build a fixture from")
        breeds = sorted(d for d in os.listdir(CROPS_TRAIN)
                        if os.path.isdir(os.path.join(CROPS_TRAIN, d)))[:2]
        if len(breeds) < 2:
            pytest.skip("need at least 2 breed folders of crops")

        root = tmp_path / "tiny_cls"
        needed = TRAIN_N + VAL_N
        for split, start, count in (("train", 0, TRAIN_N), ("val", TRAIN_N, VAL_N)):
            for breed in breeds:
                src = os.path.join(CROPS_TRAIN, breed)
                images = [f for f in sorted(os.listdir(src))
                          if f.endswith((".jpg", ".jpeg", ".png"))]
                if len(images) < needed:
                    pytest.skip(f"breed {breed} has fewer than {needed} crops for a split")
                dst = root / split / breed
                dst.mkdir(parents=True)
                for f in images[start:start + count]:
                    shutil.copy2(os.path.join(src, f), dst / f)
        return str(root), breeds

    def test_tiny_two_class_training_produces_loadable_weights(self, tiny_dataset, tmp_path):
        from ultralytics import YOLO

        from bovid.predict import load_model, predict

        data_dir, breeds = tiny_dataset
        model = YOLO(config.CLASSIFIER_MODEL)
        model.train(data=data_dir, epochs=1, imgsz=64, batch=4, device=config.DEVICE,
                    project=str(tmp_path / "runs"), name="tiny", verbose=False, plots=False,
                    cache=False)

        best = tmp_path / "runs" / "tiny" / "weights" / "best.pt"
        assert best.exists(), "training did not write a best.pt"

        val = os.path.join(data_dir, "val")
        breed_dir = os.path.join(val, sorted(os.listdir(val))[0])
        any_crop = os.path.join(breed_dir, sorted(os.listdir(breed_dir))[0])

        result = predict(any_crop, model=load_model(str(best)))
        assert result.top.breed in breeds
        assert 0.0 <= result.top.confidence <= 1.0
