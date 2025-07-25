import os
import cv2
import numpy as np
import requests
from typing import List, Dict


class ObjectDetector:
    """Real-time object detector powered by MobileNet-SSD.

    Automatically downloads the Caffe model on first use and can leverage
    CUDA when available.
    """

    PROTOTXT_URL: str = (
        "https://gist.githubusercontent.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f/raw/3d2765b625f1b090669a05d0b3e79b2907677e86/"
        "MobileNetSSD_deploy.prototxt"
    )
    MODEL_URL: str = (
        "https://github.com/robmarkcole/object-detection-app/raw/refs/heads/master/model/"
        "MobileNetSSD_deploy.caffemodel"
    )

    CLASSES: List[str] = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(
        self,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        use_gpu: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        self._setup_model()

    # ---------------------------------------------------------------------
    # Model helpers
    # ---------------------------------------------------------------------

    @property
    def _model_dir(self) -> str:
        d: str = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(d, exist_ok=True)
        return d

    def _download(self, url: str, dst: str) -> None:
        print(f"[INFO] Downloading {url} → {dst}")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(dst, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r   {pct:6.2f}%", end="")
        print("\n[INFO] Download complete.")

    def _ensure_model_files(self) -> tuple[str, str]:
        proto = os.path.join(self._model_dir, "MobileNetSSD_deploy.prototxt")
        model = os.path.join(self._model_dir, "MobileNetSSD_deploy.caffemodel")
        if not os.path.isfile(proto):
            self._download(self.PROTOTXT_URL, proto)
        if not os.path.isfile(model):
            self._download(self.MODEL_URL, model)
        return proto, model

    def _setup_model(self) -> None:
        proto, model = self._ensure_model_files()
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] Using CUDA backend for detection")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on a BGR image and return list of detections."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        boxes, confidences, class_ids = [], [], []
        for i in range(outputs.shape[2]):
            conf = float(outputs[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            idx = int(outputs[0, 0, i, 1])
            if idx >= len(self.CLASSES):
                continue
            box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            boxes.append([startX, startY, endX - startX, endY - startY])
            confidences.append(conf)
            class_ids.append(idx)

        # NMS to remove duplicates
        keep = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections: List[Dict] = []
        if len(keep) > 0:
            for j in keep.flatten():
                detections.append(
                    {
                        "id": j,
                        "class_name": self.CLASSES[class_ids[j]],
                        "confidence": float(confidences[j]),
                        "bbox": boxes[j],  # [x, y, w, h]
                    }
                )
        return detections 