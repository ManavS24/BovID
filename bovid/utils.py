"""Shared image-IO helpers. Images are loaded EXIF-corrected and passed around as BGR ndarrays
(the crop/detector convention); conversion to RGB/PIL happens only at the edges.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_rgb(path):
    """Open an image as an EXIF-corrected RGB PIL image (phone photos carry an orientation tag)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image_bgr(path):
    """Open an image as an EXIF-corrected BGR ndarray (what the crop/detector stage expects)."""
    return cv2.cvtColor(np.array(load_image_rgb(path)), cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr):
    """Convert a BGR ndarray back to an RGB PIL image (for saving or classifier input)."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
