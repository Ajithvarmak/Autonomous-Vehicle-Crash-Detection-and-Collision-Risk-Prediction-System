from __future__ import annotations

import os
import threading
import time
from typing import List

import cv2
import numpy as np

import config
from distance_estimator import Status, VehicleResult

# AudioAlert

class AudioAlert:
    """
    Thread-safe audio alert with consecutive-frame confirmation and cooldown.

    Call  update(True/False)  once per processed frame.
    The sound fires only when the streak reaches ALERT_CONFIRM_FRAMES AND
    at least ALERT_COOLDOWN_SEC seconds have elapsed since the last alert.
    """

    def __init__(self) -> None:
        self._streak      = 0
        self._last_fired  = 0.0
        self._lock        = threading.Lock()
        self._mixer_ready = self._init_mixer()


    def update(self, alert_condition: bool) -> None:
        """Call once per frame with whether an alert condition exists."""
        with self._lock:
            if alert_condition:
                self._streak += 1
                ready = (
                    self._streak >= config.ALERT_CONFIRM_FRAMES
                    and time.time() - self._last_fired >= config.ALERT_COOLDOWN_SEC
                )
                if ready:
                    self._last_fired = time.time()
                    threading.Thread(target=self._play, daemon=True).start()
            else:
                self._streak = 0

    def _init_mixer(self) -> bool:
        try:
            import pygame
            pygame.mixer.init()
            return True
        except Exception as exc:
            print(f"[AudioAlert] pygame.mixer unavailable – audio disabled. ({exc})")
            return False

    def _play(self) -> None:
        if not self._mixer_ready:
            return
        path = config.ALERT_SOUND_PATH
        if not os.path.exists(path):
            print(f"[AudioAlert] Sound file not found: {path}")
            return
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as exc:
            print(f"[AudioAlert] Playback error: {exc}")

# VisualRenderer

class VisualRenderer:
    """
    Stateless frame renderer.  All methods mutate the frame in-place (OpenCV
    convention) and return nothing.

    Public entry-point:
        renderer.draw(frame, vehicles, system_status, fps, frame_count)
    """

    # ── Public ───────────────────────────────────────────────────────────────

    def draw(
        self,
        frame:         np.ndarray,
        vehicles:      List[VehicleResult],
        system_status: Status,
        fps:           float,
        frame_count:   int,
    ) -> None:
        """Render all overlays onto *frame* in-place."""
        is_accident = system_status == Status.ACCIDENT

        # Draw order: boxes → HUD → border (border always on top)
        for v in vehicles:
            self._draw_vehicle(frame, v)

        self._draw_hud(frame, vehicles, system_status, fps, frame_count)

        if is_accident:
            self._draw_accident_border(frame)

    def _draw_vehicle(self, frame: np.ndarray, v: VehicleResult) -> None:
        """
        Draw a colour-coded bounding box with corner brackets and a label badge.

        Visual language
        ───────────────
          SAFETY  →  thin green corner brackets, no fill
          CLOSE   →  medium yellow solid rectangle
          DANGER  →  thick red solid rectangle + red-tinted fill
          ACCIDENT→  thick deep-red rectangle + darker fill + pulsing
        """
        color = v.status.color
        x1, y1, x2, y2 = v.x1, v.y1, v.x2, v.y2
        bl = self._bracket_len(v)   # corner bracket length

        if v.status in (Status.DANGER, Status.ACCIDENT):
            # Semi-transparent danger fill
            overlay = frame.copy()
            fill    = (0, 0, 60) if v.status == Status.ACCIDENT else (0, 0, 50)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            self._corners(frame, x1, y1, x2, y2, color, 4, bl)

        elif v.status == Status.CLOSE:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._corners(frame, x1, y1, x2, y2, color, 3, bl)

        else:  # SAFETY
            self._corners(frame, x1, y1, x2, y2, color, 2, bl)

        self._draw_badge(frame, v, color)

    def _draw_badge(self, frame: np.ndarray, v: VehicleResult, color: tuple) -> None:
        """Two-line label badge drawn above the bounding box."""
        # Line 1:  STATUS ICON  class_name
        icons    = {Status.SAFETY: ">>", Status.CLOSE: "!", Status.DANGER: "!!", Status.ACCIDENT: "SOS"}
        icon     = icons.get(v.status, "")
        line1    = f"{icon} {v.status.value}  [{v.class_name}]"

        # Line 2:  dist: X.Xm   conf: 0.XX
        parts = []
        if config.SHOW_DISTANCE_LABEL:
            parts.append(f"dist:{v.distance_m:.1f}m")
        if config.SHOW_CONF_LABEL:
            parts.append(f"conf:{v.conf:.2f}")
        line2 = "  ".join(parts)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs1, fs2 = 0.48, 0.40
        th1, th2 = (2 if v.status != Status.SAFETY else 1), 1

        (w1, h1), _ = cv2.getTextSize(line1, font, fs1, th1)
        (w2, h2), _ = cv2.getTextSize(line2, font, fs2, th2)

        bw = max(w1, w2) + 12
        bh = h1 + h2 + 14
        bx = v.x1
        by = max(v.y1 - bh, 0)

        # Badge background
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), config.COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 1)

        # Text
        cv2.putText(frame, line1, (bx + 5, by + h1 + 4),    font, fs1, color,            th1, cv2.LINE_AA)
        cv2.putText(frame, line2, (bx + 5, by + h1 + h2 + 10), font, fs2, config.COLOR_GRAY, th2, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame:         np.ndarray,
        vehicles:      List[VehicleResult],
        system_status: Status,
        fps:           float,
        frame_count:   int,
    ) -> None:
        h, w = frame.shape[:2]
        bar_h = 56
        cv2.rectangle(frame, (0, 0), (w, bar_h), config.COLOR_HUD_BG, -1)

        # Animated dot for ACCIDENT
        if system_status == Status.ACCIDENT:
            pulse_r = int(abs(np.sin(time.time() * 5)) * 8) + 6
            cv2.circle(frame, (18, 28), pulse_r, config.COLOR_ACCIDENT, -1)
            prefix = "   "
        else:
            prefix = ""

        status_text  = f"{prefix}{system_status.value}"
        status_color = system_status.color
        cv2.putText(frame, status_text, (34, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, status_color, 2, cv2.LINE_AA)

        # Divider line under bar
        cv2.line(frame, (0, bar_h), (w, bar_h), (40, 40, 40), 1)

        counts = {
            Status.ACCIDENT: sum(1 for v in vehicles if v.status == Status.ACCIDENT),
            Status.DANGER:   sum(1 for v in vehicles if v.status == Status.DANGER),
            Status.CLOSE:    sum(1 for v in vehicles if v.status == Status.CLOSE),
            Status.SAFETY:   sum(1 for v in vehicles if v.status == Status.SAFETY),
        }
        labels = [
            (f"ACC:{counts[Status.ACCIDENT]}", config.COLOR_ACCIDENT),
            (f"DGR:{counts[Status.DANGER]}",   config.COLOR_DANGER),
            (f"CLO:{counts[Status.CLOSE]}",    config.COLOR_CLOSE),
            (f"SAF:{counts[Status.SAFETY]}",   config.COLOR_SAFE),
        ]
        cx = w - 8
        for text, color in reversed(labels):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cx -= tw + 14
            cv2.putText(frame, text, (cx, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        items = [
            ("SAFETY",   config.COLOR_SAFE),
            ("CLOSE",    config.COLOR_CLOSE),
            ("DANGER",   config.COLOR_DANGER),
            ("ACCIDENT", config.COLOR_ACCIDENT),
        ]
        ly = h - 78
        for label, color in items:
            cv2.rectangle(frame, (8, ly - 11), (21, ly + 2), color, -1)
            cv2.putText(frame, label, (28, ly + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            ly += 18

        strip_y = h - 26
        cv2.rectangle(frame, (0, strip_y), (w, h), config.COLOR_HUD_BG, -1)
        cv2.line(frame, (0, strip_y), (w, strip_y), (40, 40, 40), 1)

        info_parts = [f"Vehicles: {len(vehicles)}"]
        if config.SHOW_FPS:
            info_parts.append(f"FPS: {fps:.1f}")
        info_parts.append(f"Frame: {frame_count}")
        cv2.putText(frame, "  |  ".join(info_parts), (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 255, 130), 1, cv2.LINE_AA)

        ts = time.strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, ts, (w - tw - 10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 255, 130), 1, cv2.LINE_AA)

    def _draw_accident_border(self, frame: np.ndarray) -> None:
        """Full-frame pulsing red border – unmistakable visual alarm."""
        t         = time.time()
        thickness = int(abs(np.sin(t * 5)) * 4) + 3     # oscillates 3–7 px
        h, w      = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), config.COLOR_ACCIDENT, thickness)

    @staticmethod
    def _bracket_len(v: VehicleResult) -> int:
        return max(12, min(v.width, v.height) // 5)

    @staticmethod
    def _corners(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple, thickness: int, length: int,
    ) -> None:
        """Draw four L-shaped corner accents."""
        segs = [
            ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
        ]
        for p1, p2 in segs:
            cv2.line(frame, p1, p2, color, thickness)
