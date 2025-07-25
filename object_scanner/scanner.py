from __future__ import annotations

import time
from typing import List, Dict, Tuple
import cv2

from .detector import ObjectDetector
from .tracker import ObjectTracker


class ObjectScanner:
    """High-level real-time object scanner (detection + optional tracking)."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        detection_interval: int = 5,
        enable_tracking: bool = True,
    ) -> None:
        self.detector: ObjectDetector = detector or ObjectDetector()
        self.detection_interval = max(1, detection_interval)
        self.enable_tracking = enable_tracking
        self.tracker: ObjectTracker | None = ObjectTracker() if enable_tracking else None
        self._frame_idx: int = 0
        self._prev_ts: float = time.time()
        self.fps: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        now = time.time()
        diff = now - self._prev_ts
        self.fps = 1.0 / diff if diff else 0.0
        self._prev_ts = now

    def _draw_overlay(self, frame, detections: List[Dict]) -> None:
        for det in detections:
            x, y, w, h = det["bbox"]
            label = f"{det['class_name']}:{det['id']}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            frame,
            f"FPS: {self.fps:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 220, 50),
            2,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame) -> Tuple:
        """Process a frame, returning (annotated_frame, detections)."""
        run_detection = self._frame_idx % self.detection_interval == 0 or not self.enable_tracking
        if run_detection:
            detections = self.detector.detect(frame)
            if self.enable_tracking and self.tracker:
                self.tracker.init_tracks(frame, detections)
        else:
            detections = self.tracker.update(frame) if self.tracker else []

        self._draw_overlay(frame, detections)
        self._update_fps()
        self._frame_idx += 1
        return frame, detections 