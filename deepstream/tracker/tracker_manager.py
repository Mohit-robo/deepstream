"""
TrackerManager — Multi-Object Lifecycle Manager

Manages a pool of TrackerInstance objects sharing one SUTrackEngine.
Handles initialization, per-frame update, IoU-based detection matching,
and stale tracker removal.

Tracker lifecycle:
1. Detector (or manual init) provides [x, y, w, h] → TrackerManager.initialize()
2. Each frame: TrackerManager.update(frame_rgb, detections, frame_idx)
   - Runs SUTrack for each active tracker
   - Optionally matches new detections to existing tracks via IoU
   - Spawns new TrackerInstance for unmatched high-confidence detections
3. Trackers unseen for > max_age frames → removed
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from .sutrack_engine import SUTrackEngine
from .tracker_instance import TrackerInstance
from .tracker_utils import compute_iou

logger = logging.getLogger(__name__)


class TrackerManager:
    """
    Multi-object tracker lifecycle manager.

    Args:
        engine:              Shared SUTrackEngine instance
        max_age:             Frames a tracker can go unmatched before deletion (default 30)
        min_confidence:      Minimum detector confidence to spawn a new tracker (default 0.25)
        iou_match_threshold: Minimum IoU to match a detection to an existing track (default 0.3)
        template_factor:     Crop factor for template extraction (default 2.0)
        search_factor:       Crop factor for search extraction (default 4.0)
        use_hanning:         Apply Hanning window during tracking (default True)
    """

    def __init__(self, engine: SUTrackEngine,
                 max_age: int = 30,
                 min_confidence: float = 0.25,
                 iou_match_threshold: float = 0.3,
                 template_factor: float = 2.0,
                 search_factor: float = 4.0,
                 use_hanning: bool = True):
        self.engine               = engine
        self.max_age              = max_age
        self.min_confidence       = min_confidence
        self.iou_match_threshold  = iou_match_threshold
        self.template_factor      = template_factor
        self.search_factor        = search_factor
        self.use_hanning          = use_hanning

        self._next_id: int = 0
        self.active_trackers: Dict[int, TrackerInstance] = {}

    # ------------------------------------------------------------------
    def initialize(self, frame_rgb: np.ndarray, bbox: list,
                   frame_idx: int = 0) -> int:
        """
        Manually initialize a new tracker for a given bounding box.

        Args:
            frame_rgb:  H×W×3 uint8 RGB image
            bbox:       [x, y, w, h] pixel bounding box
            frame_idx:  Current frame number

        Returns:
            Assigned object_id for the new tracker
        """
        obj_id = self._next_id
        self._next_id += 1

        inst = TrackerInstance(
            object_id=obj_id,
            template_factor=self.template_factor,
            search_factor=self.search_factor,
            template_size=self.engine.template_size,
            search_size=self.engine.search_size,
            feat_sz=self.engine.feat_sz,
            use_hanning=self.use_hanning,
        )
        inst.initialize(frame_rgb, bbox, frame_idx)
        self.active_trackers[obj_id] = inst
        logger.info('Initialized tracker id=%d  bbox=%s  frame=%d', obj_id, bbox, frame_idx)
        return obj_id

    # ------------------------------------------------------------------
    def update(self, frame_rgb: np.ndarray,
               frame_idx: int,
               detections: Optional[List[dict]] = None) -> Dict[int, list]:
        """
        Run one tracking step for all active trackers, then optionally match
        new detections and spawn trackers for unmatched ones.

        Args:
            frame_rgb:   H×W×3 uint8 RGB image (must be RGB — Lesson 12)
            frame_idx:   Current frame number
            detections:  Optional list of dicts with keys 'bbox' ([x,y,w,h])
                         and 'confidence' (float). Pass None to skip matching.

        Returns:
            dict mapping object_id → [x, y, w, h] for all active trackers
        """
        results: Dict[int, list] = {}

        # Run SUTrack for every active tracker (sequential — shared engine)
        for obj_id, inst in list(self.active_trackers.items()):
            try:
                bbox = inst.track(frame_rgb, self.engine, frame_idx)
                results[obj_id] = bbox
            except Exception as e:
                logger.warning('Tracker id=%d raised exception: %s — removing', obj_id, e)
                del self.active_trackers[obj_id]

        # Match detections to existing tracks and spawn new ones
        if detections:
            matched_det_indices = set()
            tracked_ids = list(results.keys())

            # Greedy IoU matching: for each existing track find best detection
            for obj_id in tracked_ids:
                best_iou, best_det_idx = 0.0, -1
                for i, det in enumerate(detections):
                    if i in matched_det_indices:
                        continue
                    iou = compute_iou(results[obj_id], det['bbox'])
                    if iou > best_iou:
                        best_iou, best_det_idx = iou, i

                if best_iou >= self.iou_match_threshold and best_det_idx >= 0:
                    matched_det_indices.add(best_det_idx)
                    # Update last_seen so the tracker is not pruned
                    self.active_trackers[obj_id].last_seen_frame = frame_idx
                    logger.debug('Track id=%d matched detection %d (IoU=%.2f)',
                                 obj_id, best_det_idx, best_iou)

            # Spawn new trackers for unmatched high-confidence detections
            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                if det.get('confidence', 1.0) >= self.min_confidence:
                    new_id = self.initialize(frame_rgb, det['bbox'], frame_idx)
                    results[new_id] = det['bbox']
                    logger.info('Spawned tracker id=%d from detection (conf=%.2f)',
                                new_id, det.get('confidence', 1.0))

        self.remove_stale(frame_idx)
        return results

    # ------------------------------------------------------------------
    def remove_stale(self, current_frame: int):
        """Remove trackers that have not been updated for > max_age frames."""
        stale = [oid for oid, inst in self.active_trackers.items()
                 if (current_frame - inst.last_seen_frame) > self.max_age]
        for oid in stale:
            logger.info('Removing stale tracker id=%d (last seen frame %d)',
                        oid, self.active_trackers[oid].last_seen_frame)
            del self.active_trackers[oid]

    # ------------------------------------------------------------------
    @property
    def num_active(self) -> int:
        return len(self.active_trackers)
