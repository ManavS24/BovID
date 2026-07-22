"""Unified `bovid` command-line interface.

    bovid predict IMAGE [IMAGE ...]   classify one or more photos
    bovid evaluate                    accuracy on the held-out test split
    bovid train                       run the full training pipeline
    bovid export                      regenerate deployment artifacts (ONNX/TFLite/SavedModel)
    bovid audit                       scan the dataset for baked-in watermarks/overlays

Diagnostics go to stderr via logging; command results go to stdout, so output can be piped
(`bovid predict img.jpg > result.txt`) without log noise.
"""

import argparse
import logging

from . import config
from .logging_conf import setup_logging

logger = logging.getLogger(__name__)


def _cmd_predict(args):
    # Imported inside the branch so --edge never pulls in ultralytics/torch.
    if args.edge:
        from .edge import EdgePipeline

        results = EdgePipeline().predict_batch(args.image, topk=args.topk)
    else:
        from .predict import load_model, predict_batch

        model = load_model(args.model)
        results = predict_batch(args.image, model=model, imgsz=args.imgsz, topk=args.topk)

    for result in results:
        print(f"\nTop-{args.topk} predictions for {result.image_path}  [crop: {result.crop_method}]")
        for breed, confidence in result.predictions:
            print(f" - {breed}: {confidence:.3f}")
        if not result.is_confident:
            print(f"   note: top-1 below {config.CONFIDENCE_THRESHOLD:.0%} — treat as uncertain.")


def _cmd_evaluate(args):
    from .evaluate import main as evaluate_main

    evaluate_main()


def _cmd_train(args):
    from .train import main as train_main

    train_main()


def _cmd_export(args):
    from .export import EXPORT_FORMATS, export_model

    produced = export_model(weights=args.model, formats=args.formats, imgsz=args.imgsz,
                            out_dir=args.out_dir)
    print("\nExported artifacts:")
    for name in (args.formats or list(EXPORT_FORMATS)):
        path = produced.get(name)
        print(f" - {name:12s}: {path if path else 'unavailable (see log)'}")


def _cmd_audit(args):
    from .audit import main as audit_main

    audit_main()


def build_parser():
    parser = argparse.ArgumentParser(prog="bovid", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="classify one or more images")
    p.add_argument("image", nargs="+", help="path(s) to image file(s)")
    p.add_argument("--model", default=config.DEFAULT_MODEL_PATH, help="model weights")
    p.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    p.add_argument("--topk", type=int, default=config.DEFAULT_TOPK)
    p.add_argument("--edge", action="store_true",
                   help="use the torch-free TFLite pipeline (what runs on a device)")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("evaluate", help="accuracy on the held-out test split")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("train", help="run the full training pipeline")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("export", help="regenerate deployment artifacts")
    p.add_argument("--model", default=None, help="weights to export (default: deployed best.pt)")
    p.add_argument("--formats", nargs="+", default=None,
                   help="subset of: onnx tflite_fp32 tflite_fp16 tflite_int8 saved_model")
    p.add_argument("--imgsz", type=int, default=None)
    # Needed to export a detector into models/pretrained/ rather than the classifier dir.
    p.add_argument("--out-dir", default=None,
                   help="where to place artifacts (default: the deployed model dir)")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("audit", help="scan the dataset for baked-in watermarks/overlays")
    p.set_defaults(func=_cmd_audit)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else None)
    args.func(args)


if __name__ == "__main__":
    main()
