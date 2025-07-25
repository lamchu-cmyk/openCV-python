from __future__ import annotations

from typing import List, Dict
import cv2


class _Track:
    def __init__(self, tracker: cv2.Tracker, track_id: int, label: str):
        self.tracker = tracker
        self.id = track_id
        self.label = label


class ObjectTracker:
    """Lightweight multi-object tracker leveraging OpenCV KCF trackers."""

    def __init__(self) -> None:
        self._tracks: List[_Track] = []
        self._next_id: int = 0

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _create_tracker(self) -> cv2.Tracker:
        # Try different tracker APIs in order of preference and availability
        # This handles compatibility across different OpenCV versions
        
        # Try CSRT (most accurate)
        try:
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
        except:
            pass
            
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        except:
            pass
        
        # Try KCF (good balance of speed and accuracy)
        try:
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
        except:
            pass
            
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()
        except:
            pass
        
        # Try MOSSE (fastest)
        try:
            if hasattr(cv2, "TrackerMOSSE_create"):
                return cv2.TrackerMOSSE_create()
        except:
            pass
            
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                return cv2.legacy.TrackerMOSSE_create()
        except:
            pass
        
        # Try newer tracker APIs if available
        try:
            # Some newer OpenCV versions use different constructors
            return cv2.TrackerCSRT.create()
        except:
            pass
            
        try:
            return cv2.TrackerKCF.create()
        except:
            pass
            
        try:
            return cv2.TrackerMOSSE.create()
        except:
            pass
        
        # If all else fails, raise an informative error
        raise RuntimeError(
            "No compatible OpenCV tracker found. Please check your OpenCV installation. "
            "Consider installing opencv-contrib-python if you haven't already."
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def init_tracks(self, frame, detections: List[Dict]) -> None:
        """(Re)initialise trackers from detector output."""
        self.reset()
        for det in detections:
            x, y, w, h = det["bbox"]
            tracker = self._create_tracker()
            tracker.init(frame, (x, y, w, h))
            self._tracks.append(_Track(tracker, self._next_id, det["class_name"]))
            self._next_id += 1

    def update(self, frame) -> List[Dict]:
        """Update trackers; return current bounding boxes."""
        valid: List[Dict] = []
        for tr in self._tracks:
            ok, box = tr.tracker.update(frame)
            if not ok:
                continue
            x, y, w, h = map(int, box)
            valid.append(
                {
                    "id": tr.id,
                    "class_name": tr.label,
                    "confidence": 1.0,
                    "bbox": [x, y, w, h],
                }
            )
        return valid 