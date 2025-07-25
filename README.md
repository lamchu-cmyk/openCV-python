# Real-Time Object Scanner

A robust, real-time object scanner built with Python, OpenCV, and NumPy. It provides high-accuracy object detection, optional multi-object tracking, and is optimized for minimal latency across CPU and GPU hardware.

## Features

* **MobileNet-SSD Detector** – fast and lightweight, suitable for real-time use on CPUs.
* **Automatic Model Download** – the required Caffe model is downloaded on first use.
* **Optional GPU Acceleration** – seamlessly switch to CUDA when OpenCV is built with GPU support.
* **Multi-Object Tracking** – object IDs persist between detection cycles using OpenCV trackers.
* **Configurable Detection Interval** – balance accuracy vs. speed by tuning how often full detection runs.
* **Modular API** – `ObjectDetector`, `ObjectTracker`, and `ObjectScanner` classes can be used independently or together.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you compiled OpenCV with CUDA, enable GPU acceleration via `--use-gpu` flag.

## Quick Start

```bash
python examples/scan_webcam.py --camera 0 --interval 5 --use-gpu
```

Press **q** to quit the demo window.

## Library Usage

```python
import cv2
from object_scanner import ObjectScanner

cap = cv2.VideoCapture(0)
scanner = ObjectScanner(detection_interval=5, enable_tracking=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame, detections = scanner.process(frame)
    # ... your application logic ...
```

## Performance Tips

1. **Adjust `detection_interval`** – increasing the interval reduces detector load; the tracker keeps objects persistent.
2. **Frame Size** – processing smaller frames increases FPS. Resize before passing to `process()` if high resolution is not required.
3. **GPU Support** – compile OpenCV with `-D WITH_CUDA=ON` and run the demo with `--use-gpu`.
4. **Batch Processing** – for multi-camera setups instantiate multiple `ObjectScanner` instances, or share one detector across threads.

## Supported Classes

The MobileNet-SSD model recognises 20 categories: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, and tvmonitor.

## License

This project is released under the MIT License. 

## Model files note

The first time you run the scanner the `ObjectDetector` will automatically
fetch the MobileNet-SSD Caffe model (\~25 MB) and its accompanying prototxt.
The downloader now tries several mirrors (original repo, OpenCV 3rd-party
mirror, alternative branch) and will keep going until one succeeds. If **all**
mirrors are unreachable (some corporate networks block `raw.githubusercontent.com`)
you can download the two files manually and drop them into
`object_scanner/models/`:

```
MobileNetSSD_deploy.prototxt
MobileNetSSD_deploy.caffemodel
```

You can obtain them from any of the following sources:

1. https://github.com/chuanqi305/MobileNet-SSD (root of repo)
2. https://github.com/opencv/opencv_3rdparty (path: dnn/...)

Once the files are present the downloader will skip the network step. 