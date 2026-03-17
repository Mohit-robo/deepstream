"""
TrackerInstance — Single-Object Tracker State

Holds template buffers and bounding-box state for one tracked object.
Implements initialize() and track() using the shared SUTrackEngine.

Post-processing rules enforced here (from tasks/lessons.md):
- Lesson 3: No double sigmoid — score_map/size_map already activated in TRT graph
- Lesson 4: Position-only clamp — never recalculate w/h from clamped corners
- Lesson 5: CPU-heavy ops avoided — pure NumPy throughout
- Lesson 6: Hanning window applied BEFORE argmax
"""

import logging
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from .tracker_utils import (
    preprocess, sample_target, transform_image_to_crop, hann2d
)

logger = logging.getLogger(__name__)


class TrackerInstance:
    """
    State container and inference logic for one tracked object.

    Args:
        object_id:      Unique integer ID for this track
        template_factor: Search area factor used for template crop (default 2.0)
        search_factor:   Search area factor used for search crop (default 4.0)
        template_size:   Template crop output size in pixels (default 112)
        search_size:     Search crop output size in pixels (default 224)
        feat_sz:         Feature map side length = search_size // encoder_stride (default 14)
        use_hanning:     Whether to apply the Hanning spatial penalty (default True)
    """

    def __init__(self, object_id: int,
                 template_factor: float = 2.0,
                 search_factor: float = 4.0,
                 template_size: int = 112,
                 search_size: int = 224,
                 feat_sz: int = 14,
                 use_hanning: bool = True):
        self.object_id       = object_id
        self.template_factor = template_factor
        self.search_factor   = search_factor
        self.template_size   = template_size
        self.search_size     = search_size
        self.feat_sz         = feat_sz

        self.state: list          = None   # [x, y, w, h] in image pixels
        self._template_img:  np.ndarray = None  # (1, 6, template_size, template_size)
        self._template_anno: np.ndarray = None  # (1, 4)
        self.last_seen_frame: int = 0
        self.confidence: float    = 1.0

        self.hann_window: np.ndarray = None
        if use_hanning:
            self.hann_window = hann2d(feat_sz, feat_sz, centered=True)

    # ------------------------------------------------------------------
    def initialize(self, frame_rgb: np.ndarray, bbox: list, frame_idx: int = 0):
        """
        Prepare template buffers from the initialization frame.

        Args:
            frame_rgb:  H×W×3 uint8 RGB image (Lesson 12: must be RGB, not BGR)
            bbox:       [x, y, w, h] pixel bounding box of the target
            frame_idx:  Current frame number (for stale tracking)
        """
        self.state = list(bbox)
        self.last_seen_frame = frame_idx

        z_crop, rf = sample_target(frame_rgb, self.state,
                                   self.template_factor, self.template_size)
        self._template_img = preprocess(z_crop)  # (1, 6, 112, 112)

        anno = transform_image_to_crop(self.state, self.state, rf, self.template_size)
        self._template_anno = anno[np.newaxis].astype(np.float32)  # (1, 4)

        logger.debug('TrackerInstance %d initialized  bbox=%s', self.object_id, self.state)

    # ------------------------------------------------------------------
    def track(self, frame_rgb: np.ndarray, engine, frame_idx: int = 0) -> list:
        """
        Run one tracking step using the shared SUTrackEngine.

        Args:
            frame_rgb:  H×W×3 uint8 RGB image (must be RGB, not BGR — Lesson 12)
            engine:     SUTrackEngine shared instance
            frame_idx:  Current frame number

        Returns:
            [x, y, w, h] updated bounding box in pixel coordinates
        """
        H, W = frame_rgb.shape[:2]

        x_crop, resize_factor = sample_target(
            frame_rgb, self.state, self.search_factor, self.search_size)
        search_np = preprocess(x_crop)  # (1, 6, 224, 224)

        trt_out = engine.infer(self._template_img, search_np, self._template_anno)

        # --- Post-processing (pure NumPy) ---
        # score_map and size_map are already sigmoid-activated inside the TRT graph.
        # Do NOT apply sigmoid again (Lesson 3).
        score_map  = trt_out['score_map'].reshape(self.feat_sz, self.feat_sz)
        size_map   = trt_out['size_map'].reshape(2, self.feat_sz * self.feat_sz)
        offset_map = trt_out['offset_map'].reshape(2, self.feat_sz * self.feat_sz)

        # Apply Hanning window BEFORE argmax — spatial suppression of distractors (Lesson 6)
        if self.hann_window is not None:
            score_map = score_map * self.hann_window

        score_flat = score_map.reshape(-1)
        idx   = int(np.argmax(score_flat))
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        w_norm = float(size_map[0, idx])
        h_norm = float(size_map[1, idx])
        off_x  = float(offset_map[0, idx])
        off_y  = float(offset_map[1, idx])

        cx_norm = (idx_x + off_x) / self.feat_sz
        cy_norm = (idx_y + off_y) / self.feat_sz

        # Scale from normalized → crop-space pixels
        scale = self.search_size / resize_factor
        cx = cx_norm * scale
        cy = cy_norm * scale
        w  = w_norm  * scale
        h  = h_norm  * scale

        # Map crop-space → original image coordinates
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx_real = cx + (cx_prev - 0.5 * scale)
        cy_real = cy + (cy_prev - 0.5 * scale)

        # Position-only clamp — preserve predicted w, h (Lesson 4)
        x1 = max(0.0, min(cx_real - 0.5 * w, W - 1.0))
        y1 = max(0.0, min(cy_real - 0.5 * h, H - 1.0))

        self.state = [x1, y1, w, h]
        self.last_seen_frame = frame_idx
        self.confidence = float(score_flat[idx])
        return self.state
