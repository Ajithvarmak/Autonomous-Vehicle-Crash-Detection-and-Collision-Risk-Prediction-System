
# config.py
# Accident Prediction Alert System – Central Configuration
# All tunable parameters live here. No logic code needs to be touched when
# adjusting thresholds, paths, or display settings.

VEHICLE_MODEL_PATH: str = "yolov8n.pt"

# YOLOv8 custom accident detection model (ONNX format)
ACCIDENT_MODEL_PATH: str = "models/accident.onnx"

# DETECTION SETTINGS
CONF_THRESHOLD:    float = 0.40
NMS_IOU_THRESHOLD: float = 0.45
YOLO_INFER_SIZE:   int   = 320

# COCO vehicle class IDs used by YOLOv8n
VEHICLE_CLASS_IDS: dict = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MIN_BOX_SIZE: int = 35
# VIDEO SOURCE

VIDEO_SOURCE = 0          # 0 = webcam; pass a file path string for video file
SCREEN_WIDTH:  int = 800
SCREEN_HEIGHT: int = 500
FRAME_SKIP:    int = 1    # process every Nth frame through YOLO


FOCAL_LENGTH_PX:    float = 600.0
KNOWN_CAR_HEIGHT_M: float = 1.5

CLASS_REAL_HEIGHTS: dict = {
    1: 1.0,   # bicycle
    2: 1.5,   # car
    3: 1.2,   # motorcycle
    5: 3.2,   # bus
    7: 3.0,   # truck
}

# RISK THRESHOLDS
#  distance > SAFE_DISTANCE_M           → SAFETY  (green)
#  CLOSE_DISTANCE_M < d <= SAFE         → CLOSE   (yellow)
#  distance <= CLOSE_DISTANCE_M         → DANGER  (red)

SAFE_DISTANCE_M:  float = 15.0
CLOSE_DISTANCE_M: float = 7.0

ACCIDENT_CONF_THRESHOLD: float = 0.65   # accident model min confidence
ALERT_CONFIRM_FRAMES:    int   = 3      # consecutive frames before audio fires

# ALERT AUDIO
ALERT_SOUND_PATH:   str   = "audio/warning.mp3"
ALERT_COOLDOWN_SEC: float = 4.0

# COLOURS  (BGR)
COLOR_SAFE:     tuple = (34,  177,  76)
COLOR_CLOSE:    tuple = (0,   200, 230)
COLOR_DANGER:   tuple = (30,   30, 220)
COLOR_ACCIDENT: tuple = (0,     0, 200)
COLOR_HUD_BG:   tuple = (18,   18,  18)
COLOR_WHITE:    tuple = (245, 245, 245)
COLOR_GRAY:     tuple = (160, 160, 160)

# DISPLAY FLAGS
SHOW_DISTANCE_LABEL: bool = True
SHOW_CONF_LABEL:     bool = True
SHOW_FPS:            bool = True