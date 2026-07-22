"""Synthetic watermark/overlay + real-world-corruption augmentation for overlay robustness.

Stamps random text overlays (mimicking the contamination the audit found) onto TRAIN crops only,
so the classifier learns overlays carry no breed signal; val/test keep their real watermarks.
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Brand-like / caption-like strings echoing the real overlays seen in the dataset.
PHRASES = [
    "getty images", "shutterstock", "istockphoto", "alamy stock photo", "dreamstime",
    "© 2024", "royalty free", "stock photo", "sample", "preview", "HD 1080p",
    "subscribe now", "like & share", "wonderful life", "www.example.com", "123rf",
]


def _font(size):
    return ImageFont.load_default(size=size)


def _random_text(rng):
    if rng.random() < 0.2:  # a video timestamp like "1:20 / 3:28"
        return f"{rng.randint(0, 9)}:{rng.randint(0, 59):02d} / {rng.randint(1, 9)}:{rng.randint(0, 59):02d}"
    return rng.choice(PHRASES)


def _draw_text_layer(text, font, fill):
    """Render text onto its own tight RGBA layer (so it can be rotated/positioned freely)."""
    dummy = Image.new("RGBA", (1, 1))
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    layer = Image.new("RGBA", (max(w, 1) + 8, max(h, 1) + 8), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((4 - box[0], 4 - box[1]), text, font=font, fill=fill)
    return layer


def add_watermark(img_rgb, rng=None):
    """Return a copy of img_rgb (H,W,3 uint8) with one random synthetic watermark stamped on."""
    rng = rng or random
    base = Image.fromarray(img_rgb).convert("RGBA")
    W, H = base.size

    style = rng.choice(["bottom", "corner", "center_diag"])
    text = _random_text(rng)
    color = rng.choice([(255, 255, 255), (235, 235, 235), (0, 0, 0), (200, 200, 200)])

    if style == "center_diag":
        # large, low-opacity, rotated -- like a Getty/Shutterstock center watermark
        alpha = rng.randint(35, 90)
        font = _font(int(H * rng.uniform(0.09, 0.14)))
        layer = _draw_text_layer(text, font, color + (alpha,))
        layer = layer.rotate(rng.uniform(-35, -10), expand=True)
        pos = ((W - layer.width) // 2, (H - layer.height) // 2)
    elif style == "corner":
        alpha = rng.randint(120, 210)
        font = _font(int(H * rng.uniform(0.04, 0.07)))
        layer = _draw_text_layer(text, font, color + (alpha,))
        m = int(W * 0.03)
        corner = rng.choice(["tl", "tr", "bl", "br"])
        x = m if "l" in corner else W - layer.width - m
        y = m if "t" in corner else H - layer.height - m
        pos = (x, y)
    else:  # bottom caption bar
        alpha = rng.randint(150, 230)
        font = _font(int(H * rng.uniform(0.05, 0.08)))
        layer = _draw_text_layer(text, font, color + (alpha,))
        pos = ((W - layer.width) // 2, H - layer.height - int(H * 0.02))

    base.alpha_composite(layer, pos)
    return np.array(base.convert("RGB"))


def add_motion_blur(img_rgb, rng):
    """Directional blur -- phone photos of moving animals / shaky hands. randaugment doesn't do this."""
    size = rng.choice([7, 9, 11, 13])
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img_rgb, -1, kernel)


def add_jpeg_artifacts(img_rgb, rng):
    """Low-quality JPEG recompression -- messaging-app / low-bandwidth uploads."""
    q = rng.randint(25, 60)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img_rgb
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def add_brightness(img_rgb, rng):
    """Under/over-exposure -- harsh sun or low light."""
    return np.clip(img_rgb.astype(np.float32) * rng.uniform(0.6, 1.4), 0, 255).astype(np.uint8)


def realworld_augment(img_rgb, rng):
    """Watermark + real-world phone-capture corruptions, each applied at random."""
    out = img_rgb
    if rng.random() < 0.6:
        out = add_watermark(out, rng)
    if rng.random() < 0.5:
        out = add_motion_blur(out, rng)
    if rng.random() < 0.5:
        out = add_jpeg_artifacts(out, rng)
    if rng.random() < 0.5:
        out = add_brightness(out, rng)
    return out


def _build_augmented_train(train_dir, aug_fn, seed=42):
    """Add one aug_<name> variant per clean train image (idempotent; never augments an aug_ file)."""
    rng = random.Random(seed)
    added = 0
    for breed_dir in sorted(p for p in Path(train_dir).iterdir() if p.is_dir()):
        originals = [f for f in sorted(breed_dir.iterdir())
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("aug_")]
        for f in originals:
            out = breed_dir / f"aug_{f.name}"
            if out.exists():
                continue
            img = np.array(Image.open(f).convert("RGB"))
            Image.fromarray(aug_fn(img, rng)).save(out, format="JPEG", quality=95)
            added += 1
    logger.info("added %d augmented train variants under %s", added, train_dir)


def build_watermarked_train(train_dir, seed=42):
    """D: one synthetic-watermark variant per clean train image (overlay robustness only)."""
    _build_augmented_train(train_dir, add_watermark, seed)


def build_realworld_train(train_dir, seed=42):
    """One real-world-corrupted variant per clean train image."""
    _build_augmented_train(train_dir, realworld_augment, seed)
