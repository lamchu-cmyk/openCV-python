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

FOR WINDOWS:
```bash
pip install -r requirements.txt
```

If you compiled OpenCV with CUDA, enable GPU acceleration via `--use-gpu` flag.

## Quick Start

```bash
python examples/scan_webcam.py --camera 0 --interval 5 --use-gpu
```

Press **Ctrl + C** to quit the demo window.
