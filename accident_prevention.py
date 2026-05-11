from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np

import config
from warning_system import AudioAlert, VisualRenderer
from distance_estimator import DistanceEstimator, Status, VehicleResult

# YOLOv8n Vehicle Detector

class YOLOv8nDetector:
    """
    Vehicle detector powered by YOLOv8n via the ultralytics Python API.

    Why ultralytics and not raw ONNX for YOLOv8?
    ─────────────────────────────────────────────
    YOLOv8's ONNX export uses a transposed output layout [1, 84, N] that
    requires non-trivial anchor decoding.  The official ultralytics library
    handles preprocessing, decoding, and NMS internally, returning clean
    (x1, y1, x2, y2, conf, class_id) boxes in two lines of code.

    CPU optimisation  (Intel i3 / 8 GB RAM)
    ────────────────────────────────────────
    • yolov8n  = nano model, smallest and fastest in the YOLOv8 family
    • imgsz=320 reduces inference time vs 640 with acceptable accuracy
    • half=False  – FP16 not supported on CPU
    • device="cpu" – explicit, no accidental GPU use
    • verbose=False – suppresses per-frame ultralytics console output
    """

    def __init__(self, model_path: str) -> None:
        print(f"[YOLOv8n] Loading model: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[YOLOv8n] ultralytics not installed.  Run: pip install ultralytics")
            sys.exit(1)
        self._model = YOLO(model_path)
        self._model.to("cpu")
        print("[YOLOv8n] Ready  (device=cpu, imgsz=320)")

    def detect(self, frame: np.ndarray) -> List[tuple]:
        """
        Run YOLOv8n on a BGR frame.

        Returns a list of (x1, y1, x2, y2, conf, class_id) tuples for
        vehicle classes only.  Coordinates are clamped to frame boundaries.
        NMS is handled internally by ultralytics.
        """
        results = self._model.predict(
            source=frame,
            imgsz=config.YOLO_INFER_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            device="cpu",
            half=False,
            verbose=False,
        )

        fh, fw   = frame.shape[:2]
        vehicles = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            if cls not in config.VEHICLE_CLASS_IDS:
                continue
            vehicles.append((
                max(0, int(x1)), max(0, int(y1)),
                min(fw, int(x2)), min(fh, int(y2)),
                float(conf), cls,
            ))
        return vehicles

# ONNX Accident Detector

class AccidentDetector:
    """
    Binary accident classifier using an ONNX model via ONNX Runtime.

    Input  : full frame (resized to model input size)
    Output : accident confidence in [0, 1]

    The model is optional.  If the file is missing the detector degrades
    gracefully – it always returns 0.0 confidence so the rest of the system
    works without it.

    Output format handling
    ──────────────────────
    We support two common export layouts:
      • [1, 2]  – softmax binary classifier  → probs[1] = accident class
      • [1, 1]  – sigmoid single output      → value directly
      • anything else                        → np.max fallback
    """

    def __init__(self, model_path: str) -> None:
        self._session = None
        self._input_name: str  = ""
        self._input_size: int  = 640

        if not os.path.exists(model_path):
            print(f"[AccidentDetector] Model not found: {model_path} – running without accident detection.")
            return

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            self._session    = ort.InferenceSession(model_path, sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._input_size = self._session.get_inputs()[0].shape[2]   # assumes square
            print(f"[AccidentDetector] Loaded ONNX model  (input_size={self._input_size})")
        except Exception as exc:
            print(f"[AccidentDetector] Failed to load ONNX model: {exc}")

    def confidence(self, frame: np.ndarray) -> float:
        """Return accident confidence [0.0, 1.0].  Returns 0.0 if no model."""
        if self._session is None:
            return 0.0
        blob = self._preprocess(frame)
        out  = self._session.run(None, {self._input_name: blob})[0]
        return self._parse_output(out)


    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        sz  = self._input_size
        img = cv2.resize(frame, (sz, sz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

    @staticmethod
    def _parse_output(out: np.ndarray) -> float:
        arr = out[0]  # shape varies by model
        if arr.ndim == 1 and arr.shape[0] == 2:
            # Softmax binary: [no_accident, accident]
            exp   = np.exp(arr - arr.max())
            probs = exp / exp.sum()
            return float(probs[1])
        if arr.ndim == 1 and arr.shape[0] == 1:
            # Sigmoid single output
            return float(1.0 / (1.0 + np.exp(-arr[0])))
        # Fallback
        return float(np.max(arr))



# Video Source
class VideoSource:
    """
    Thin cv2.VideoCapture wrapper supporting webcam index or video file.
    File sources loop automatically so the system runs continuously.
    """

    def __init__(self, source) -> None:
        self._cap     = cv2.VideoCapture(source)
        self._is_file = isinstance(source, str)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.SCREEN_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.SCREEN_HEIGHT)
        print(f"[VideoSource] Opened: {source}")

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            if self._is_file:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                return None
        return cv2.resize(frame, (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    def release(self) -> None:
        self._cap.release()


def aggregate_status(
    vehicles:     List[VehicleResult],
    acc_conf:     float,
) -> Status:
    """
    Determine the single system-wide Status from all per-vehicle statuses
    and the accident model confidence.

    Priority (highest wins):
      ACCIDENT  if acc_conf > ACCIDENT_CONF_THRESHOLD
      DANGER    if any vehicle is in DANGER
      CLOSE     if any vehicle is in CLOSE
      SAFETY    otherwise
    """
    if acc_conf >= config.ACCIDENT_CONF_THRESHOLD:
        return Status.ACCIDENT

    statuses = {v.status for v in vehicles}
    if Status.DANGER in statuses:
        return Status.DANGER
    if Status.CLOSE in statuses:
        return Status.CLOSE
    return Status.SAFETY


def run(source=config.VIDEO_SOURCE) -> None:
    """Initialise all components and start the real-time processing loop."""

    try:
        vehicle_detector  = YOLOv8nDetector(config.VEHICLE_MODEL_PATH)
    except Exception as exc:
        print(f"[main] YOLOv8n init failed: {exc}")
        sys.exit(1)

    accident_detector = AccidentDetector(config.ACCIDENT_MODEL_PATH)
    estimator         = DistanceEstimator()
    audio_alert       = AudioAlert()
    renderer          = VisualRenderer()

    try:
        video = VideoSource(source)
    except RuntimeError as exc:
        print(f"[main] {exc}")
        sys.exit(1)

    os.makedirs("screenshots", exist_ok=True)
    frame_count   = 0
    fps_history   = deque(maxlen=30)
    last_time     = time.time()
    paused        = False
    last_vehicles: List[VehicleResult] = []
    last_status   = Status.SAFETY
    screenshot_n  = 0
    banner = [
        "=" * 64,
        "  ACCIDENT PREDICTION ALERT SYSTEM  |  LIVE",
        "=" * 64,
        f"  Vehicle model  : {config.VEHICLE_MODEL_PATH}",
        f"  Accident model : {config.ACCIDENT_MODEL_PATH}",
        f"  Source         : {source}",
        f"  Resolution     : {config.SCREEN_WIDTH}×{config.SCREEN_HEIGHT}",
        f"  Safe threshold : >{config.SAFE_DISTANCE_M} m  (GREEN)",
        f"  Close threshold: >{config.CLOSE_DISTANCE_M} m  (YELLOW)",
        f"  Danger         : <={config.CLOSE_DISTANCE_M} m  (RED)",
        "-" * 64,
        "  Controls:  Q – Quit  |  P – Pause  |  R – Reset  |  S – Screenshot",
        "=" * 64,
    ]
    print("\n".join(banner))

    while True:

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("⏸  Paused" if paused else "▶  Resumed")
        elif key == ord("r"):
            frame_count = 0
            fps_history.clear()
            print("↺  Reset")
        elif key == ord("s"):
            path = f"screenshots/frame_{screenshot_n:04d}.jpg"
            # We'll save the annotated frame on the next loop iteration
            screenshot_n += 1

        if paused:
            cv2.waitKey(30)
            continue

        frame = video.read()
        if frame is None:
            print("[main] End of stream.")
            break

        frame_count += 1
        now       = time.time()
        fps       = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        fps_history.append(fps)
        avg_fps = float(np.mean(fps_history))

        if frame_count % config.FRAME_SKIP == 0:
            raw_boxes = vehicle_detector.detect(frame)

            vehicles: List[VehicleResult] = []
            for (x1, y1, x2, y2, conf, cls) in raw_boxes:
                result = estimator.process(x1, y1, x2, y2, cls, conf)
                if result is not None:
                    vehicles.append(result)

            last_vehicles = vehicles
        else:
            vehicles = last_vehicles  

        acc_conf = accident_detector.confidence(frame)

        system_status = aggregate_status(vehicles, acc_conf)
        last_status   = system_status


        alert_condition = system_status in (Status.DANGER, Status.ACCIDENT)
        audio_alert.update(alert_condition)

        renderer.draw(frame, vehicles, system_status, avg_fps, frame_count)

        if key == ord("s"):
            spath = f"screenshots/frame_{screenshot_n - 1:04d}.jpg"
            cv2.imwrite(spath, frame)
            print(f"📸  Screenshot saved: {spath}")

        cv2.imshow("Accident Prediction Alert System", frame)

    video.release()
    cv2.destroyAllWindows()
    print(f"\n[main] Session ended – {frame_count} frames processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accident Prediction Alert System")
    parser.add_argument(
        "--source",
        default=None,
        help="Webcam index (int) or path to a video file.  Default: webcam 0.",
    )
    args = parser.parse_args()

    src = args.source
    if src is not None:
        try:
            src = int(src)      
        except ValueError:
            pass                
    else:
        src = config.VIDEO_SOURCE

    run(src)