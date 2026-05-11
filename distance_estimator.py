
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import config

class Status(Enum):
    """
    System-wide status levels.

    SAFETY   – no vehicles in proximity (initial / cleared state)
    CLOSE    – vehicle detected within SAFE_DISTANCE_M
    DANGER   – vehicle within CLOSE_DISTANCE_M
    ACCIDENT – accident model fired with high confidence
    """
    SAFETY   = "SAFETY"
    CLOSE    = "CLOSE PROXIMITY"
    DANGER   = "DANGER"
    ACCIDENT = "ACCIDENT DETECTED"

    @property
    def color(self) -> tuple:
        return {
            Status.SAFETY:   config.COLOR_SAFE,
            Status.CLOSE:    config.COLOR_CLOSE,
            Status.DANGER:   config.COLOR_DANGER,
            Status.ACCIDENT: config.COLOR_ACCIDENT,
        }[self]

    @property
    def box_color(self) -> tuple:
        """Box outline colour is the same as the status colour."""
        return self.color

@dataclass
class VehicleResult:
    """All information about one detected vehicle for a single frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id:   int
    class_name: str
    conf:       float
    distance_m: float
    status:     Status

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height
    
# Distance estimator
class DistanceEstimator:
    """
    Converts a YOLO bounding box into a metric distance and a Status level.

    Usage
    ─────
        estimator = DistanceEstimator()
        result = estimator.process(x1, y1, x2, y2, class_id, conf)
        # result is VehicleResult or None if the box is too small
    """

    def __init__(self) -> None:
        self._focal   = config.FOCAL_LENGTH_PX
        self._heights = config.CLASS_REAL_HEIGHTS
        self._default = config.KNOWN_CAR_HEIGHT_M

    def process(
        self,
        x1: int, y1: int, x2: int, y2: int,
        class_id: int,
        conf: float,
    ) -> Optional[VehicleResult]:
        """
        Estimate distance and classify risk for one bounding box.

        Returns None when the box is too small to produce a reliable reading
        (both sides must be >= MIN_BOX_SIZE).
        """
        box_h = y2 - y1
        box_w = x2 - x1

        if box_h < config.MIN_BOX_SIZE or box_w < config.MIN_BOX_SIZE:
            return None

        class_name  = config.VEHICLE_CLASS_IDS.get(class_id, "vehicle")
        real_h      = self._heights.get(class_id, self._default)
        distance_m  = self._distance(box_h, real_h)
        status      = self._classify(distance_m)

        return VehicleResult(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=class_id,
            class_name=class_name,
            conf=conf,
            distance_m=distance_m,
            status=status,
        )

    def _distance(self, box_h_px: int, real_h_m: float) -> float:
        """Apply pinhole formula, clamped to [1 m, 120 m]."""
        if box_h_px <= 0:
            return 120.0
        return float(max(1.0, min((real_h_m * self._focal) / box_h_px, 120.0)))

    @staticmethod
    def _classify(d: float) -> Status:
        if d > config.SAFE_DISTANCE_M:
            return Status.SAFETY
        if d > config.CLOSE_DISTANCE_M:
            return Status.CLOSE
        return Status.DANGER
