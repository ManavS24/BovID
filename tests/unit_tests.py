"""Unit suite (one of the two entry points): every module tested in isolation, no real
weights/runtime/dataset. Shared guards live in _bovid_reqs.py."""

import importlib
import random

import numpy as np
import pytest


class TestConfig:
    @pytest.fixture(autouse=True)
    def _restore_config(self):
        yield
        from bovid import config
        importlib.reload(config)

    @staticmethod
    def _reload(monkeypatch, *, clear=(), **env):
        for key in clear:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from bovid import config
        return importlib.reload(config)

    def test_env_override_wins(self, monkeypatch, tmp_path):
        models = tmp_path / "somewhere" / "models"
        models.mkdir(parents=True)
        config = self._reload(monkeypatch, clear=("BOVID_MODELS_ROOT", "BOVID_DATA_ROOT"),
                              BOVID_MODELS_ROOT=str(models))
        assert config.MODELS_ROOT == str(models)
        assert config.PRETRAINED_DIR.startswith(str(models))
        assert config.EDGE_DETECTOR_MODEL.startswith(str(models))

    def test_falls_back_to_the_repo_checkout(self, monkeypatch):
        import os
        config = self._reload(monkeypatch, clear=("BOVID_MODELS_ROOT", "BOVID_DATA_ROOT"))
        assert config.MODELS_ROOT == os.path.join(config.PROJECT_ROOT, "models")
        assert os.path.isdir(config.MODELS_ROOT), "this test assumes it runs from a checkout"

    def test_env_override_is_absolute(self, monkeypatch, tmp_path):
        import os
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.chdir(tmp_path)
        config = self._reload(monkeypatch, clear=("BOVID_MODELS_ROOT", "BOVID_DATA_ROOT"),
                              BOVID_MODELS_ROOT="models")
        assert os.path.isabs(config.MODELS_ROOT)
        assert config.MODELS_ROOT == str(models)

    def test_valid_override_is_accepted(self, monkeypatch):
        config = self._reload(monkeypatch, BOVID_CONFIDENCE_THRESHOLD="0.8")
        assert config.CONFIDENCE_THRESHOLD == 0.8

    def test_confidence_threshold_out_of_range_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="CONFIDENCE_THRESHOLD"):
            self._reload(monkeypatch, BOVID_CONFIDENCE_THRESHOLD="1.5")

    def test_inverted_coverage_gates_are_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="DETECT_MIN_COV"):
            self._reload(monkeypatch, BOVID_DETECT_MIN_COV="0.9", BOVID_DETECT_MAX_COV="0.5")

    def test_fallback_trims_that_would_empty_the_crop_are_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="FALLBACK"):
            self._reload(monkeypatch, BOVID_FALLBACK_BOTTOM_TRIM="0.8", BOVID_FALLBACK_EDGE_TRIM="0.45")

    def test_negative_topk_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="DEFAULT_TOPK"):
            self._reload(monkeypatch, BOVID_DEFAULT_TOPK="0")

    def test_bad_device_is_rejected_on_access(self, monkeypatch):
        """DEVICE is validated lazily (to keep torch out of the edge path), so it raises on
        access, not import."""
        config = self._reload(monkeypatch, BOVID_DEVICE="gpu0")
        with pytest.raises(ValueError, match="DEVICE"):
            _ = config.DEVICE

    def test_confidence_threshold_accepts_the_range_endpoints(self, monkeypatch):
        assert self._reload(monkeypatch, BOVID_CONFIDENCE_THRESHOLD="0.0").CONFIDENCE_THRESHOLD == 0.0
        assert self._reload(monkeypatch, BOVID_CONFIDENCE_THRESHOLD="1.0").CONFIDENCE_THRESHOLD == 1.0


class TestAudit:
    @staticmethod
    def _mod():
        pytest.importorskip("pytesseract", reason="audit imports pytesseract at load time")
        import bovid.audit as audit
        return audit

    def test_empty_text_is_clean(self):
        classify = self._mod().classify
        assert classify("") is None
        assert classify(None) is None

    def test_stock_photo_keyword_flagged(self):
        assert self._mod().classify("Getty Images / 2024") == "keyword:getty"

    def test_every_declared_keyword_is_detected(self):
        audit = self._mod()
        for kw in audit.SUSPICIOUS_KEYWORDS:
            assert audit.classify(f"prefix {kw.upper()} suffix") == f"keyword:{kw}"

    def test_video_timestamp_flagged(self):
        assert self._mod().classify("1:20 / 3:28") == "video-timestamp"

    def test_generic_text_run_flagged(self):
        assert self._mod().classify("Sindhi") == "generic-text"

    def test_short_non_keyword_text_is_clean(self):
        classify = self._mod().classify
        assert classify("ab") is None
        assert classify("12 3") is None

    def test_keyword_takes_precedence_over_generic(self):
        assert self._mod().classify("subscribe to the channel").startswith("keyword:")

    def test_generic_text_is_inclusive_at_four_letters(self):
        classify = self._mod().classify
        assert classify("abc") is None
        assert classify("abcd") == "generic-text"

    def test_whitespace_only_text_is_clean(self):
        assert self._mod().classify("   \n\t ") is None

    def test_url_fragment_inside_a_word_is_flagged(self):
        assert self._mod().classify("visit example.com today") == "keyword:.com"

    @staticmethod
    def _make_images(dir_path, names):
        for name in names:
            (dir_path / name).write_bytes(b"\xff\xd8\xff")
        return dir_path

    def test_scan_split_globs_supported_extensions_sorted(self, tmp_path):
        import os
        scan_split = self._mod().scan_split
        self._make_images(tmp_path, ["b.jpg", "a.png", "c.jpeg", "notes.txt", "d.gif"])
        image_paths, _ = scan_split(str(tmp_path), ocr=lambda p: "")
        assert [os.path.basename(str(p)) for p in image_paths] == ["a.png", "b.jpg", "c.jpeg"]

    def test_scan_split_flags_only_contaminated_images(self, tmp_path):
        import os
        scan_split = self._mod().scan_split
        self._make_images(tmp_path, ["clean.jpg", "stock.jpg"])
        texts = {"clean.jpg": "", "stock.jpg": "Getty Images"}
        _, flagged = scan_split(str(tmp_path), ocr=lambda p: texts[os.path.basename(str(p))])
        assert len(flagged) == 1
        path, reason, _snippet = flagged[0]
        assert os.path.basename(path) == "stock.jpg" and reason == "keyword:getty"

    def test_scan_split_truncates_and_flattens_the_snippet(self, tmp_path):
        scan_split = self._mod().scan_split
        self._make_images(tmp_path, ["x.jpg"])
        _, flagged = scan_split(str(tmp_path), ocr=lambda p: "getty\n" + "a" * 200)
        snippet = flagged[0][2]
        assert len(snippet) == 80 and "\n" not in snippet

    def test_scan_split_on_empty_dir_returns_empty(self, tmp_path):
        paths, flagged = self._mod().scan_split(str(tmp_path), ocr=lambda p: "getty")
        assert paths == [] and flagged == []


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)


_AUG_FNS = pytest.mark.parametrize("fn_name", [
    "add_watermark", "add_motion_blur", "add_jpeg_artifacts", "add_brightness", "realworld_augment",
])


class TestAugment:
    @_AUG_FNS
    def test_augmentation_is_deterministic_under_a_fixed_seed(self, fn_name, image):
        from bovid import augment
        fn = getattr(augment, fn_name)
        np.testing.assert_array_equal(fn(image, random.Random(42)), fn(image, random.Random(42)))

    @_AUG_FNS
    def test_augmentation_preserves_shape_and_dtype(self, fn_name, image):
        from bovid import augment
        out = getattr(augment, fn_name)(image, random.Random(1))
        assert out.shape == image.shape and out.dtype == np.uint8

    def test_different_seeds_usually_differ(self, image):
        from bovid import augment
        outputs = [augment.realworld_augment(image, random.Random(s)).tobytes() for s in range(5)]
        assert len(set(outputs)) > 1

    def test_watermark_actually_changes_the_image(self, image):
        from bovid import augment
        assert not np.array_equal(augment.add_watermark(image, random.Random(3)), image)

    def test_brightness_stays_in_range(self, image):
        from bovid import augment
        out = augment.add_brightness(image, random.Random(7))
        assert out.min() >= 0 and out.max() <= 255


class _FakeDetector:
    """Records the kwargs it was called with; stands in for a YOLO model."""

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)

        class _R:
            names = {0: "cow"}
            boxes = []

        return [_R()]


class _Box:
    def __init__(self, xyxy, cls):
        self.xyxy = [xyxy]
        self.cls = cls


class _BoxResult:
    def __init__(self, boxes):
        self.names = {0: "cow", 1: "person"}
        self.boxes = boxes


class TestCrop:
    @pytest.mark.parametrize("ckpt,expect_device", [
        ("models/pretrained/yolov8x.pt", True),
        ("models/pretrained/yolov8n_float16.tflite", False),
        ("models/best/best.onnx", False),
    ])
    def test_device_passed_only_to_torch_backends(self, ckpt, expect_device):
        """device= to a TFLite/ONNX backend raises 'Unsupported device type: mps'."""
        from bovid import config, crop
        detector = _FakeDetector(ckpt)
        crop._best_animal_box(np.zeros((64, 64, 3), dtype=np.uint8), detector)
        kwargs = detector.calls[0]
        assert ("device" in kwargs) is expect_device
        assert kwargs["conf"] == config.DETECT_CONF

    def test_fallback_crop_is_square_and_trims_the_bottom_margin(self):
        from bovid import crop
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[-40:, :] = 255
        out = crop._deterministic_crop(img)
        assert out.shape[0] == out.shape[1], "fallback crop must be square"
        assert out.max() == 0, "bottom margin (where overlays live) should have been trimmed away"

    def test_tiny_box_rejected_by_coverage_gate(self):
        from bovid import crop
        tiny = _BoxResult([_Box((0, 0, 10, 10), 0)])
        assert crop._box_from_result(tiny, 400, 400) is None
        big = _BoxResult([_Box((10, 10, 390, 390), 0)])
        assert crop._box_from_result(big, 400, 400) == (10, 10, 390, 390)

    def test_non_animal_class_ignored(self):
        from bovid import crop
        person = _BoxResult([_Box((10, 10, 390, 390), 1)])
        assert crop._box_from_result(person, 400, 400) is None

    def test_no_box_takes_the_fallback_path(self):
        from bovid import crop
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        assert crop.crop_from_box(img, None)[1] == "fallback"
        assert crop.crop_from_box(img, (10, 10, 390, 390))[1] == "detector"

    def test_degenerate_box_below_the_min_size_falls_back(self):
        from bovid import crop
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        assert crop.crop_from_box(img, (10, 10, 20, 20))[1] == "fallback"   # 10x10 < 32px floor

    def test_padding_expands_top_left_right_but_not_bottom(self):
        from bovid import config, crop
        assert config.DETECT_BOTTOM_PAD == 0.0, "test assumes the shipped default"
        x1, y1, x2, y2 = crop._pad_box(100, 100, 200, 200, w=1000, h=1000)
        assert x1 < 100 and y1 < 100 and x2 > 200
        assert y2 == 200, "bottom edge must NOT move down -- that is where watermarks live"

    def test_padding_is_clamped_to_image_bounds(self):
        from bovid import crop
        x1, y1, x2, y2 = crop._pad_box(0, 0, 300, 300, w=300, h=300)
        assert x1 >= 0 and y1 >= 0 and x2 <= 300 and y2 <= 300

    def test_padding_amount_matches_the_configured_fraction(self):
        from bovid import config, crop
        x1, _y1, x2, _y2 = crop._pad_box(100, 100, 300, 300, w=10_000, h=10_000)
        expected = int(200 * config.DETECT_PAD)
        assert 100 - x1 == expected and x2 - 300 == expected

    def test_box_from_result_with_no_boxes_returns_none(self):
        from bovid import crop
        assert crop._box_from_result(_BoxResult([]), 400, 400) is None


class TestSourceGroup:
    def test_augmentation_siblings_share_a_group(self):
        from bovid.dataset import source_group
        siblings = [
            "00000002_jpg.rf.1ee756fbfe553855d3c9ea3202474cd3.jpg",
            "00000002_jpg.rf.3e67b41c2bdb93bc8ee2507c33d2eb0c.jpg",
            "00000002_jpg.rf.4b30423ec4d1da506dcecdf094657f46.jpg",
        ]
        assert {source_group(n) for n in siblings} == {"00000002_jpg"}

    def test_different_source_photos_are_different_groups(self):
        from bovid.dataset import source_group
        assert source_group("00000002_jpg.rf.aaaa.jpg") != source_group("00000133_jpg.rf.bbbb.jpg")

    def test_filename_without_rf_marker_is_its_own_group(self):
        from bovid.dataset import source_group
        assert source_group("plain_image.jpg") == "plain_image.jpg"

    def test_group_key_is_deterministic(self):
        from bovid.dataset import source_group
        assert source_group("00000002_jpg.rf.deadbeef.jpg") == source_group("00000002_jpg.rf.deadbeef.jpg")

    def test_only_the_first_rf_marker_splits(self):
        from bovid.dataset import source_group
        assert source_group("a_jpg.rf.b.rf.c.jpg") == "a_jpg"


class TestTopkMetrics:
    def test_all_correct(self):
        from bovid.evaluate import topk_metrics
        m = topk_metrics([
            ("Gir", ["Gir", "Sahiwal", "Rathi"], 0.9),
            ("Ongole", ["Ongole", "Nagori", "Kankrej"], 0.8),
        ])
        assert m["n"] == 2 and m["top1_acc"] == 1.0 and m["top3_acc"] == 1.0
        assert m["correct_conf"] == [0.9, 0.8] and m["incorrect_conf"] == []

    def test_top3_hit_that_is_not_a_top1_hit(self):
        from bovid.evaluate import topk_metrics
        m = topk_metrics([("Gir", ["Sahiwal", "Rathi", "Gir"], 0.55)])
        assert m["top1_acc"] == 0.0 and m["top3_acc"] == 1.0
        assert m["incorrect_conf"] == [0.55] and m["correct_conf"] == []

    def test_complete_miss(self):
        from bovid.evaluate import topk_metrics
        m = topk_metrics([("Gir", ["Sahiwal", "Rathi", "Nagori"], 0.6)])
        assert m["top1_acc"] == 0.0 and m["top3_acc"] == 0.0 and m["incorrect_conf"] == [0.6]

    def test_mixed_batch_confidence_routing(self):
        from bovid.evaluate import topk_metrics
        m = topk_metrics([
            ("Gir", ["Gir", "Sahiwal", "Rathi"], 0.95),
            ("Ongole", ["Nagori", "Ongole", "Kankrej"], 0.40),
            ("Rathi", ["Sahiwal", "Nagori", "Kankrej"], 0.70),
        ])
        assert m["top1_acc"] == pytest.approx(1 / 3) and m["top3_acc"] == pytest.approx(2 / 3)
        assert m["correct_conf"] == [0.95] and sorted(m["incorrect_conf"]) == [0.40, 0.70]

    def test_empty_records_is_an_error_not_a_zero_division(self):
        from bovid.evaluate import topk_metrics
        with pytest.raises(ValueError, match="no records"):
            topk_metrics([])

    def test_single_candidate_list_still_scores(self):
        from bovid.evaluate import topk_metrics
        m = topk_metrics([("Gir", ["Gir"], 0.9)])
        assert m["top1_acc"] == 1.0 and m["top3_acc"] == 1.0


class TestExport:
    @staticmethod
    def _mod():
        pytest.importorskip("ultralytics", reason="export imports ultralytics at load time")
        import bovid.export as export
        return export

    @staticmethod
    def _touch(path):
        with open(path, "w") as f:
            f.write("x")
        return str(path)

    def test_resolve_int8_finds_onnx2tf_alias(self, tmp_path):
        export = self._mod()
        self._touch(tmp_path / "best_full_integer_quant.tflite")
        assert export._resolve_int8(str(tmp_path / "best_int8.tflite")) == \
            str(tmp_path / "best_full_integer_quant.tflite")

    def test_resolve_int8_returns_none_when_absent(self, tmp_path):
        export = self._mod()
        assert export._resolve_int8(str(tmp_path / "best_int8.tflite")) is None

    def test_copy_into_none_and_missing_return_none(self, tmp_path):
        export = self._mod()
        assert export._copy_into(None, str(tmp_path)) is None
        assert export._copy_into(str(tmp_path / "nope.tflite"), str(tmp_path)) is None

    def test_copy_into_copies_a_real_file(self, tmp_path):
        import os
        export = self._mod()
        src = self._touch(tmp_path / "best.onnx")
        dest = tmp_path / "out"
        dest.mkdir()
        result = export._copy_into(src, str(dest))
        assert result == str(dest / "best.onnx") and os.path.exists(result)

    def test_copy_into_is_a_noop_when_already_in_place(self, tmp_path):
        import os
        export = self._mod()
        src = self._touch(tmp_path / "best.pt")
        result = export._copy_into(src, str(tmp_path))
        assert result == src and os.path.exists(result)

    def test_copy_into_resolves_int8_alias(self, tmp_path):
        import os
        export = self._mod()
        self._touch(tmp_path / "best_integer_quant.tflite")
        dest = tmp_path / "out"
        dest.mkdir()
        result = export._copy_into(str(tmp_path / "best_int8.tflite"), str(dest))
        assert result == str(dest / "best_integer_quant.tflite") and os.path.exists(result)

    def test_export_model_rejects_unknown_format(self, tmp_path):
        export = self._mod()
        with pytest.raises(ValueError, match="unknown export format"):
            export.export_model(formats=["bogus"], out_dir=str(tmp_path))

    def test_export_model_attempts_every_format_and_wires_paths(self, tmp_path):
        import os
        export = self._mod()
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        model = _FakeExportModel(str(src))
        produced = export.export_model(formats=["onnx", "tflite_fp32"], out_dir=str(out), model=model)
        assert model.calls == 2 and set(produced) == {"onnx", "tflite_fp32"}
        for path in produced.values():
            assert path is not None and os.path.exists(path) and os.path.dirname(path) == str(out)

    def test_export_model_isolates_a_failing_backend(self, tmp_path):
        import os
        export = self._mod()
        out = tmp_path / "out"
        model = _FakeExportModel(str(tmp_path), fail={"onnx"})
        produced = export.export_model(formats=["onnx", "tflite_fp32"], out_dir=str(out), model=model)
        assert produced["onnx"] is None
        assert produced["tflite_fp32"] is not None and os.path.exists(produced["tflite_fp32"])


class _FakeExportModel:
    """Stands in for YOLO: .export() writes a stub and returns its path, or raises for `fail`."""

    def __init__(self, work_dir, fail=()):
        import os
        self._os = os
        self.work_dir = work_dir
        self.fail = set(fail)
        self.calls = 0

    def export(self, **kwargs):
        self.calls += 1
        fmt = kwargs["format"]
        if fmt in self.fail:
            raise RuntimeError(f"{fmt} backend unavailable")
        path = self._os.path.join(self.work_dir, f"stub.{fmt}")
        with open(path, "w") as f:
            f.write("x")
        return path


class TestLoggingConf:
    def test_returns_the_bovid_logger(self):
        from bovid.logging_conf import setup_logging
        assert setup_logging().name == "bovid"

    def test_silences_ultralytics_to_warning(self):
        import logging

        from bovid.logging_conf import setup_logging
        setup_logging()
        assert logging.getLogger("ultralytics").level == logging.WARNING

    @staticmethod
    def _captured_level(monkeypatch, *, arg=None, env=None):
        """basicConfig is a no-op once handlers exist (as under pytest), by design -- so spy on it
        to assert the resolved level rather than reading the root logger."""
        from bovid import logging_conf
        if env is None:
            monkeypatch.delenv("BOVID_LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("BOVID_LOG_LEVEL", env)
        seen = {}
        monkeypatch.setattr(logging_conf.logging, "basicConfig", lambda **kw: seen.update(kw))
        logging_conf.setup_logging(arg)
        return seen["level"]

    def test_default_level_is_info(self, monkeypatch):
        assert self._captured_level(monkeypatch) == "INFO"

    def test_explicit_level_wins(self, monkeypatch):
        assert self._captured_level(monkeypatch, arg="DEBUG") == "DEBUG"

    def test_level_overridable_via_env(self, monkeypatch):
        assert self._captured_level(monkeypatch, env="warning") == "WARNING"


def _result(predictions):
    from bovid.result import Prediction, PredictionResult
    return PredictionResult(image_path="x.jpg",
                            predictions=[Prediction(b, c) for b, c in predictions],
                            crop_method="detector")


class TestResult:
    def test_top_is_the_first_prediction(self):
        from bovid.result import Prediction
        result = _result([("Gir", 0.7), ("Sahiwal", 0.2), ("Rathi", 0.1)])
        assert result.top == Prediction("Gir", 0.7) and result.top.breed == "Gir"

    def test_is_confident_true_above_threshold(self):
        from bovid import config
        assert _result([("Gir", config.CONFIDENCE_THRESHOLD + 0.01)]).is_confident is True

    def test_is_confident_false_below_threshold(self):
        from bovid import config
        assert _result([("Gir", config.CONFIDENCE_THRESHOLD - 0.01)]).is_confident is False

    def test_is_confident_is_inclusive_at_the_threshold(self):
        from bovid import config
        assert _result([("Gir", config.CONFIDENCE_THRESHOLD)]).is_confident is True

    def test_prediction_unpacks_as_a_tuple(self):
        from bovid.result import Prediction
        breed, confidence = Prediction("Ongole", 0.42)
        assert breed == "Ongole" and confidence == 0.42

    def test_result_is_frozen(self):
        from dataclasses import FrozenInstanceError
        result = _result([("Gir", 0.9)])
        with pytest.raises(FrozenInstanceError):
            result.crop_method = "fallback"


class TestTFLiteMetadata:
    @staticmethod
    def _mod():
        pytest.importorskip("yaml", reason="tflite_backend imports PyYAML at load time")
        import bovid.tflite_backend as tb
        return tb

    @pytest.mark.parametrize("filename, expected", [
        ("best_float32.tflite", "best"),
        ("best_float16.tflite", "best"),
        ("best_int8.tflite", "best"),
        ("best_full_integer_quant.tflite", "best"),
        ("best_integer_quant.tflite", "best"),
        ("best.tflite", "best"),
        ("yolov8s_cls_int8.tflite", "yolov8s_cls"),
    ])
    def test_base_stem_strips_only_the_export_suffix(self, filename, expected):
        assert self._mod()._base_stem(filename) == expected

    def test_find_metadata_prefers_the_saved_model_dir(self, tmp_path):
        find = self._mod()._find_metadata
        saved = tmp_path / "best_saved_model"
        saved.mkdir()
        (saved / "metadata.yaml").write_text("names: {0: Gir}\n")
        assert find(str(tmp_path / "best_float32.tflite")) == str(saved / "metadata.yaml")

    def test_find_metadata_falls_back_to_a_sibling(self, tmp_path):
        find = self._mod()._find_metadata
        (tmp_path / "metadata.yaml").write_text("names: {0: Gir}\n")
        assert find(str(tmp_path / "best_int8.tflite")) == str(tmp_path / "metadata.yaml")

    def test_find_metadata_locates_an_underscored_base(self, tmp_path):
        find = self._mod()._find_metadata
        saved = tmp_path / "yolov8s_cls_saved_model"
        saved.mkdir()
        (saved / "metadata.yaml").write_text("names: {0: Gir}\n")
        assert find(str(tmp_path / "yolov8s_cls_int8.tflite")) == str(saved / "metadata.yaml")

    def test_find_metadata_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.yaml"):
            self._mod()._find_metadata(str(tmp_path / "best_int8.tflite"))


class TestEdgeGeometry:
    def test_letterbox_wide_image_pads_top_bottom(self):
        from bovid.tflite_detector import letterbox
        padded, scale, pad_x, pad_y = letterbox(np.zeros((400, 800, 3), dtype=np.uint8), 640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 800)
        assert pad_y > 0 and pad_x == 0
        assert (padded[0, 0] == 114).all()          # Ultralytics' pad colour

    def test_letterbox_tall_image_pads_left_right(self):
        from bovid.tflite_detector import letterbox
        padded, scale, pad_x, pad_y = letterbox(np.zeros((800, 400, 3), dtype=np.uint8), 640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 800)
        assert pad_x > 0 and pad_y == 0

    def test_letterbox_square_image_needs_no_padding(self):
        from bovid.tflite_detector import letterbox
        _padded, scale, pad_x, pad_y = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
        assert scale == pytest.approx(1.0) and pad_x == 0 and pad_y == 0

    def test_nms_suppresses_overlapping_boxes(self):
        from bovid.tflite_detector import nms
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], dtype=float)
        keep = nms(boxes, np.array([0.9, 0.8, 0.7]), iou_threshold=0.7)
        assert keep[0] == 0 and 1 not in keep and 2 in keep

    def test_nms_on_empty_input_returns_empty(self):
        from bovid.tflite_detector import nms
        keep = nms(np.empty((0, 4)), np.empty((0,)), iou_threshold=0.5)
        assert list(keep) == []


class TestTrainHelpers:
    @staticmethod
    def _mod():
        pytest.importorskip("ultralytics", reason="train imports ultralytics at load time")
        import bovid.train as train
        return train

    def test_check_source_splits_passes_when_all_present(self, tmp_path):
        for split in ("train", "valid", "test"):
            (tmp_path / split).mkdir()
        self._mod().check_source_splits(str(tmp_path))   # must not raise

    def test_check_source_splits_raises_on_missing_split(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "valid").mkdir()
        with pytest.raises(FileNotFoundError, match="test"):
            self._mod().check_source_splits(str(tmp_path))

    def test_mb_none_for_missing_or_empty_path(self, tmp_path):
        mb = self._mod().mb
        assert mb(None) is None
        assert mb(str(tmp_path / "nope")) is None

    def test_mb_reports_size_in_megabytes(self, tmp_path):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\0" * (2 * 1024 * 1024))
        assert self._mod().mb(str(f)) == 2.0


def _write_image(path, rgb_array, exif=None):
    from PIL import Image
    img = Image.fromarray(rgb_array)
    kwargs = {"exif": exif} if exif is not None else {}
    img.save(path, format="JPEG", quality=100, **kwargs)
    return str(path)


class TestUtils:
    def test_load_image_bgr_returns_bgr_channel_order(self, tmp_path):
        from bovid.utils import load_image_bgr
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        bgr = load_image_bgr(_write_image(tmp_path / "red.jpg", rgb))
        assert bgr.shape == (16, 16, 3)
        b, _g, r = bgr[8, 8]
        assert r > 200 and b < 50, f"channels look swapped: BGR pixel = {(b, _g, r)}"

    def test_bgr_to_pil_round_trips_channels(self, tmp_path):
        from bovid.utils import bgr_to_pil, load_image_bgr
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        rgb[:, :, 2] = 40
        restored = np.array(bgr_to_pil(load_image_bgr(_write_image(tmp_path / "c.jpg", rgb))))
        np.testing.assert_allclose(restored[8, 8], rgb[8, 8], atol=3)

    def test_exif_orientation_is_applied(self, tmp_path):
        from PIL import Image

        from bovid.utils import load_image_rgb
        tall = np.zeros((40, 20, 3), dtype=np.uint8)
        tall[:20, :, 1] = 255
        exif = Image.Exif()
        exif[274] = 6                                   # tag 274 = Orientation, 6 = rotate 90 CW
        loaded = np.array(load_image_rgb(_write_image(tmp_path / "r.jpg", tall, exif=exif.tobytes())))
        assert loaded.shape[:2] == (20, 40), f"EXIF orientation not applied: got {loaded.shape[:2]}"

    def test_greyscale_is_converted_to_three_channels(self, tmp_path):
        from PIL import Image

        from bovid.utils import load_image_bgr, load_image_rgb
        path = str(tmp_path / "grey.jpg")
        Image.fromarray(np.full((16, 16), 128, dtype=np.uint8), mode="L").save(path)
        assert np.array(load_image_rgb(path)).shape == (16, 16, 3)
        assert load_image_bgr(path).shape == (16, 16, 3)
