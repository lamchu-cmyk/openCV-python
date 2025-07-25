#!/usr/bin/env python3
"""Example: real-time object scanning from webcam."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
from object_scanner import ObjectScanner


def parse_args():
    ap = argparse.ArgumentParser(description="Object Scanner webcam demo")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument(
        "--interval", type=int, default=5, help="Full detection interval in frames"
    )
    ap.add_argument("--use-gpu", action="store_true", help="Use CUDA backend if built")
    return ap.parse_args()


def main():
    args = parse_args()

    scanner = ObjectScanner(detection_interval=args.interval, enable_tracking=True)
    if args.use_gpu:
        scanner.detector.use_gpu = True
        scanner.detector._setup_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Unable to open webcam")

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out, _ = scanner.process(frame)
        cv2.imshow("Object Scanner", frame_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 