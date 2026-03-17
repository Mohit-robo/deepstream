"""
SUTrack TensorRT Deployment — Utility Functions
Pure NumPy/OpenCV implementations. Zero PyTorch dependency.
"""

import math
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2 as cv
import yaml
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Config loader (replaces lib.test.parameter.sutrack + lib.config system)
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> SimpleNamespace:
    """Load a SUTrack YAML experiment config into a plain namespace."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    cfg = _to_ns(raw)
    return cfg


class TrackerConfig:
    """Minimal tracker params extracted from a YAML config."""
    def __init__(self, yaml_path: str):
        cfg = load_config(yaml_path)
        self.cfg = cfg
        self.template_factor  = cfg.TEST.TEMPLATE_FACTOR
        self.template_size    = cfg.TEST.TEMPLATE_SIZE
        self.search_factor    = cfg.TEST.SEARCH_FACTOR
        self.search_size      = cfg.TEST.SEARCH_SIZE
        self.use_window       = cfg.TEST.WINDOW
        self.encoder_stride   = cfg.MODEL.ENCODER.STRIDE
        self.search_data_size = cfg.DATA.SEARCH.SIZE


# ---------------------------------------------------------------------------
# Image preprocessing  (replaces lib.test.tracker.utils.Preprocessor)
# ---------------------------------------------------------------------------

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize an H×W×C uint8 RGB image to a (1, C*2, H, W) float32 array
    ready for TensorRT (duplicated channels for 6-ch multi-modal input).

    Args:
        img_rgb: numpy array of shape (H, W, 3), dtype uint8, RGB order

    Returns:
        numpy array of shape (1, 6, H, W), dtype float32, C-contiguous
    """
    img = img_rgb.astype(np.float32) / 255.0          # [0,1]
    img = (img - _MEAN) / _STD                         # ImageNet normalize
    img = img.transpose(2, 0, 1)                       # (3, H, W)
    img6 = np.concatenate([img, img], axis=0)          # (6, H, W)  multi-modal stub
    return np.ascontiguousarray(img6[np.newaxis], dtype=np.float32)  # (1,6,H,W)


# ---------------------------------------------------------------------------
# Crop extraction  (replaces lib.test.tracker.utils.sample_target)
# ---------------------------------------------------------------------------

def sample_target(im: np.ndarray, target_bb: list, search_area_factor: float,
                  output_sz: int) -> tuple:
    """
    Extract a square crop centered on target_bb, padded and resized.

    Args:
        im:                 H×W×3 numpy image (any dtype)
        target_bb:          [x, y, w, h] bounding box in pixels
        search_area_factor: crop side = sqrt(w*h) * search_area_factor
        output_sz:          resize crop to this square size

    Returns:
        (crop_img, resize_factor):
            crop_img     – (output_sz, output_sz, 3) uint8 image
            resize_factor – output_sz / crop_sz  (used for coord mapping)
    """
    x, y, w, h = target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise ValueError(f'Bounding box too small: {target_bb}')

    cx = x + 0.5 * w
    cy = y + 0.5 * h
    x1 = round(cx - crop_sz * 0.5)
    y1 = round(cy - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    H_im, W_im = im.shape[:2]
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - W_im, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - H_im, 0)

    # Crop
    crop = im[max(y1,0):min(y2, H_im), max(x1,0):min(x2, W_im)]
    # Pad with zeros
    crop = cv.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # Resize
    crop = cv.resize(crop, (output_sz, output_sz))
    resize_factor = output_sz / crop_sz
    return crop, resize_factor


# ---------------------------------------------------------------------------
# Template annotation  (replaces lib.test.tracker.utils.transform_image_to_crop)
# ---------------------------------------------------------------------------

def transform_image_to_crop(box_in: list, box_extract: list,
                             resize_factor: float, crop_sz: int) -> np.ndarray:
    """
    Transform bounding box coordinates from original image space to
    normalized crop space [0,1].

    Args:
        box_in:       [x, y, w, h] box to transform (usually the init bbox)
        box_extract:  [x, y, w, h] box the crop was centered on
        resize_factor: output_sz / crop_sz
        crop_sz:      output crop size in pixels (e.g. 112)

    Returns:
        numpy array of shape (4,): [x1_n, y1_n, w_n, h_n] in [0,1]
    """
    x_in, y_in, w_in, h_in = box_in
    x_ex, y_ex, w_ex, h_ex = box_extract

    # Center of the extraction box in original image
    cx_ex = x_ex + 0.5 * w_ex
    cy_ex = y_ex + 0.5 * h_ex

    # Center of box_in in original image
    cx_in = x_in + 0.5 * w_in
    cy_in = y_in + 0.5 * h_in

    # Center of the annotation in the cropped image (pixels)
    out_cx = (crop_sz - 1) / 2.0 + (cx_in - cx_ex) * resize_factor
    out_cy = (crop_sz - 1) / 2.0 + (cy_in - cy_ex) * resize_factor
    out_w  = w_in * resize_factor
    out_h  = h_in * resize_factor

    # [x1, y1, w, h] in crop pixels
    out_x1 = out_cx - 0.5 * out_w
    out_y1 = out_cy - 0.5 * out_h

    # Normalize by (crop_sz - 1) to [0,1]
    norm = float(crop_sz - 1)
    return np.array([out_x1 / norm, out_y1 / norm, out_w / norm, out_h / norm],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Hanning window  (replaces lib.test.utils.hann.hann2d)
# ---------------------------------------------------------------------------

def hann1d(sz: int, centered: bool = True) -> np.ndarray:
    """1-D centered cosine (Hanning) window."""
    if centered:
        n = np.arange(1, sz + 1, dtype=np.float32)
        return 0.5 * (1.0 - np.cos(2.0 * math.pi * n / (sz + 1)))
    else:
        n = np.arange(0, sz // 2 + 1, dtype=np.float32)
        w = 0.5 * (1.0 + np.cos(2.0 * math.pi * n / (sz + 2)))
        return np.concatenate([w, w[1: sz - sz // 2][::-1]])


def hann2d(h: int, w: int, centered: bool = True) -> np.ndarray:
    """2-D separable Hanning window, shape (h, w)."""
    return hann1d(h, centered)[:, np.newaxis] * hann1d(w, centered)[np.newaxis, :]
