from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import threading
import math

# ================= UPGRADED IMPORTS =================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email_alert import save_accident_image, send_email_alert
from sms_alert import send_sms_alert

from twilio.rest import Client
from ultralytics import YOLO 
import cv2
import pygame
import os
from pathlib import Path
import logging
from collections import deque
import numpy as np
import json


def send_accident_whatsapp(location: str = "Unknown Location", severity: str = "CRITICAL") -> None:
    """Enhanced WhatsApp alert with severity information and threading support."""
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Configure enhanced logging with file output (UTF-8 encoding to fix emoji errors)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'detection_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import sys
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass


class AnimationController:
    """Handles all animations for the detection system."""
    
    def __init__(self):
        self.frame_count = 0
        self.alert_animation_frames = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        
    def update(self):
        """Update animation states."""
        self.frame_count += 1
        
        # Pulse animation
        self.pulse_alpha += self.pulse_direction * 0.05
        if self.pulse_alpha >= 1.0:
            self.pulse_alpha = 1.0
            self.pulse_direction = -1
        elif self.pulse_alpha <= 0.3:
            self.pulse_alpha = 0.3
            self.pulse_direction = 1
            
        # Alert animation countdown
        if self.alert_animation_frames > 0:
            self.alert_animation_frames -= 1
    
    def start_alert_animation(self):
        """Trigger alert animation."""
        self.alert_animation_frames = 60  # 60 frames animation
    
    def draw_pulse_circle(self, frame, center, radius, color):
        """Draw pulsing circle animation."""
        alpha = self.pulse_alpha
        overlay = frame.copy()
        cv2.circle(overlay, center, int(radius * alpha), color, -1)
        cv2.addWeighted(overlay, alpha * 0.3, frame, 1 - alpha * 0.3, 0, frame)
        cv2.circle(frame, center, int(radius * alpha), color, 2)
    
    def draw_warning_flash(self, frame):
        """Draw flashing warning border."""
        if self.alert_animation_frames > 0:
            h, w = frame.shape[:2]
            thickness = 10
            
            # Flash effect based on frame count
            if (self.alert_animation_frames // 5) % 2 == 0:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw border
            cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    
    def draw_radar_sweep(self, frame, center, max_radius):
        """Draw radar sweep animation."""
        angle = (self.frame_count * 3) % 360
        angle_rad = math.radians(angle)
        
        end_x = int(center[0] + max_radius * math.cos(angle_rad))
        end_y = int(center[1] + max_radius * math.sin(angle_rad))
        
        # Draw sweep line
        overlay = frame.copy()
        cv2.line(overlay, center, (end_x, end_y), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw fading trail
        for i in range(5):
            trail_angle = angle - i * 10
            trail_rad = math.radians(trail_angle)
            trail_x = int(center[0] + max_radius * math.cos(trail_rad))
            trail_y = int(center[1] + max_radius * math.sin(trail_rad))
            alpha = (5 - i) / 10
            overlay = frame.copy()
            cv2.line(overlay, center, (trail_x, trail_y), (0, 200, 0), 1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class YOLODetector:
    """Enhanced YOLO detector with improved tracking and analytics."""
    
    def __init__(self, model_path: str = 'best.pt', confidence_threshold: float = 0.5):
        """Initialize the YOLO detector with configuration."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        self.alarm_sound: Optional[pygame.mixer.Sound] = None
        self.alarm_cooldown = 0
        self.alarm_cooldown_frames = 30
        
        # Enhanced detection tracking
        self.detection_history = deque(maxlen=30)
        self.total_detections = 0
        self.detection_log: List[Dict] = []
        self.max_confidence = 0.0
        
        # WhatsApp alert cooldown
        self.whatsapp_cooldown = 0
        self.whatsapp_cooldown_frames = 150
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        
        # Animation controller
        self.animator = AnimationController()
        
        # Communication flags
        self.communication_sent = False
        
        # ✅ SOLUTION 2: Severity analysis tracking
        self.severity_level = "NORMAL"
        self.consecutive_detections = 0
        self.max_consecutive = 0
        
        # ✅ SOLUTION 3: Frame buffering for accurate image capture
        self.frame_buffer = deque(maxlen=10)  # Store last 10 frames
        self.detection_frame_index = -1

        logger.info(f"YOLODetector initialized with model: {model_path}")
        
    def load_alarm(self, alarm_path: str) -> bool:
        """Load alarm sound with error handling."""
        if os.path.exists(alarm_path):
            try:
                self.alarm_sound = pygame.mixer.Sound(alarm_path)
                logger.info(f"✅ Alarm sound loaded: {alarm_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load alarm: {e}")
                return False
        else:
            logger.warning(f"⚠️ Alarm sound not found: {alarm_path}")
            return False
    
    def play_alarm(self) -> None:
        """Play alarm with volume control."""
        if self.alarm_sound and self.alarm_cooldown == 0:
            self.alarm_sound.set_volume(0.7)  # 70% volume
            self.alarm_sound.play()
            self.alarm_cooldown = self.alarm_cooldown_frames
            logger.info("🔊 Alarm triggered!")
    
    def analyze_severity(self, detections: List[Dict]) -> str:
        """
        ✅ SOLUTION 2: Analyze accident severity based on detection patterns.
        Classifies accidents into MINOR, MODERATE, or CRITICAL based on:
        - Detection confidence levels
        - Number of consecutive detections
        - Maximum confidence achieved
        """
        if len(detections) == 0:
            self.consecutive_detections = 0
            return "NORMAL"
        
        self.consecutive_detections += 1
        self.max_consecutive = max(self.max_consecutive, self.consecutive_detections)
        
        # Get maximum confidence from current detections
        max_conf = max(d['confidence'] for d in detections)
        avg_conf = np.mean([d['confidence'] for d in detections])
        
        # Severity classification logic
        if max_conf >= 0.85 and self.consecutive_detections >= 5:
            severity = "CRITICAL"
        elif max_conf >= 0.70 and self.consecutive_detections >= 3:
            severity = "MODERATE"
        elif max_conf >= 0.50:
            severity = "MINOR"
        else:
            severity = "NORMAL"
        
        logger.info(f"📊 Severity Analysis: {severity} | Confidence: {max_conf:.2%} | Consecutive: {self.consecutive_detections}")
        return severity

    def send_whatsapp_alert(self, severity: str = "CRITICAL") -> None:
        """
        ✅ SOLUTION 1: Send WhatsApp alert in separate thread to prevent video freeze.
        Threading ensures continuous frame processing without blocking.
        """
        if self.whatsapp_cooldown == 0:
            # Run in separate thread to prevent blocking
            alert_thread = threading.Thread(
                target=send_accident_whatsapp,
                args=("Live Webcam Area", severity),
                daemon=True,
            )
            alert_thread.start()
            self.whatsapp_cooldown = self.whatsapp_cooldown_frames
            self.animator.start_alert_animation()  # Trigger animation
            logger.info(f"📱 WhatsApp alert thread started with {severity} severity")
    
    def send_email_sms_alerts(self, frame: np.ndarray, severity: str) -> None:
        """
        ✅ SOLUTION 1: Send email and SMS alerts in separate thread.
        ✅ SOLUTION 3: Uses buffered frame for accurate accident moment capture.
        """
        def alert_worker():
            """Worker function to send alerts without blocking main thread."""
            try:
                # Save the exact frame passed (already the best from buffer)
                image_path = save_accident_image(frame)
                logger.info(f"Accident image saved: {image_path}")
                
                # Send email with image
                email_success = send_email_alert(image_path, severity)
                
                # Send SMS
                sms_success = send_sms_alert(severity)
                
                if email_success and sms_success:
                    logger.info(f"All alerts sent successfully with {severity} severity")
                elif email_success:
                    logger.warning(f"Email sent but SMS failed for {severity} severity")
                elif sms_success:
                    logger.warning(f"SMS sent but email failed for {severity} severity")
                else:
                    logger.error(f"Both email and SMS failed for {severity} severity")
                    
            except Exception as e:
                logger.error(f"Alert sending failed: {e}")
        
        # Execute in separate thread
        alert_thread = threading.Thread(target=alert_worker, daemon=True)
        alert_thread.start()
        logger.info(f"Email/SMS alert thread started with {severity} severity")
    
    def get_best_detection_frame(self) -> np.ndarray:
        """
        ✅ SOLUTION 3: Retrieve the most accurate frame from buffer.
        Returns the frame that best captures the accident moment.
        """
        if len(self.frame_buffer) == 0:
            return None
        
        # Get most recent frame (last in buffer) for accurate accident capture
        # This ensures we capture the actual accident moment, not a past frame
        best_frame = self.frame_buffer[-1].copy()  # Use most recent frame
        logger.info(f"Frame selected from buffer (most recent) for accident capture")
        return best_frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process frame with enhanced detection tracking."""
        start_time = time.time()
        
        # ✅ SOLUTION 3: Store frame in buffer before processing
        self.frame_buffer.append(frame.copy())
        
        results = self.model(frame, verbose=False)
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence > self.confidence_threshold:
                    detection_data = {
                        'confidence': confidence,
                        'class': int(box.cls[0]),
                        'bbox': box.xyxy[0].tolist(),
                        'timestamp': datetime.now().isoformat()
                    }
                    detections.append(detection_data)
                    
                    # Track maximum confidence
                    if confidence > self.max_confidence:
                        self.max_confidence = confidence
            
            if len(boxes) > 0:
                annotated_frame = result.plot()
        
        # Update tracking metrics
        self.detection_history.append(len(detections))
        
        if len(detections) > 0:
            self.total_detections += len(detections)
            
            # ✅ SOLUTION 2: Analyze severity
            self.severity_level = self.analyze_severity(detections)
            
            self.detection_log.append({
                'count': len(detections),
                'timestamp': datetime.now().isoformat(),
                'max_conf': max(d['confidence'] for d in detections),
                'severity': self.severity_level  # ✅ Log severity
            })
            
            self.play_alarm()
            self.send_whatsapp_alert(self.severity_level)
            
            # ✅ SOLUTION 1 & 3: Send alerts in separate thread with best frame
            if not self.communication_sent:
                # Get the most recent frame from buffer (actual accident moment)
                best_frame = self.get_best_detection_frame()
                if best_frame is not None:
                    # Pass the actual frame, not the annotated one
                    self.send_email_sms_alerts(best_frame, self.severity_level)
                    self.communication_sent = True
                else:
                    logger.warning("No frame available in buffer for alert")
        else:
            # Reset severity when no detections
            self.consecutive_detections = 0
            self.severity_level = "NORMAL"
        
        # Update cooldowns
        if self.alarm_cooldown > 0:
            self.alarm_cooldown -= 1
        if self.whatsapp_cooldown > 0:
            self.whatsapp_cooldown -= 1
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Update animations
        self.animator.update()
        
        return annotated_frame, detections
    
    def add_stats_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float = 0) -> np.ndarray:
        """Enhanced stats overlay with gradient background and animations."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Create gradient background
        overlay_height = 220  # Increased for severity display
        for i in range(overlay_height):
            alpha = 0.7 * (1 - i / overlay_height)
            cv2.rectangle(overlay, (10, 10 + i), (450, 11 + i), (0, 0, 0), -1)
        
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Color coding for detection status
        status_color = (0, 0, 255) if len(detections) > 0 else (0, 255, 0)
        
        # ✅ SOLUTION 2: Severity color coding
        severity_colors = {
            "CRITICAL": (0, 0, 255),    # Red
            "MODERATE": (0, 165, 255),  # Orange
            "MINOR": (0, 255, 255),     # Yellow
            "NORMAL": (0, 255, 0)       # Green
        }
        severity_color = severity_colors.get(self.severity_level, (255, 255, 255))
        
        # Display enhanced statistics
        y_offset = 35
        line_height = 25
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Detections: {len(detections)}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Total: {self.total_detections}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += line_height
        avg_detections = np.mean(self.detection_history) if self.detection_history else 0
        cv2.putText(frame, f"Avg (30f): {avg_detections:.1f}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += line_height
        if detections:
            max_conf = max(d['confidence'] for d in detections)
            cv2.putText(frame, f"Max Conf: {max_conf:.2%}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += line_height
        # ✅ SOLUTION 2: Display severity level
        cv2.putText(frame, f"Severity: {self.severity_level}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, severity_color, 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Consecutive: {self.consecutive_detections}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_offset += line_height
        avg_proc_time = np.mean(self.processing_times) if self.processing_times else 0
        cv2.putText(frame, f"Proc Time: {avg_proc_time*1000:.1f}ms", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Animated alert status indicator
        if self.whatsapp_cooldown > 0:
            cv2.putText(frame, "Alert Message Sent", (w - 250, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add pulsing indicator circle
            self.animator.draw_pulse_circle(frame, (w - 270, 30), 15, (0, 255, 255))
        
        # Warning flash when detection active
        if len(detections) > 0:
            self.animator.draw_warning_flash(frame)
            
            # Add radar sweep in corner
            self.animator.draw_radar_sweep(frame, (w - 80, h - 80), 60)
        
        # Status indicator with severity
        status_text = f"{self.severity_level} ALERT" if len(detections) > 0 else "MONITORING"
        status_color_bg = severity_color if len(detections) > 0 else (0, 150, 0)
        
        # Animated status badge
        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (w - 220, h - 50), (w - 10, h - 10), status_color_bg, -1)
        cv2.addWeighted(badge_overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (w - 220, h - 50), (w - 10, h - 10), status_color_bg, 2)
        cv2.putText(frame, status_text, (w - 210, h - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_detection_log(self, output_path: Path) -> None:
        """Save detection log to JSON file."""
        log_file = output_path.parent / f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(log_file, 'w') as f:
                json.dump({
                    'total_detections': self.total_detections,
                    'max_confidence': self.max_confidence,
                    'max_consecutive_detections': self.max_consecutive,
                    'detection_events': self.detection_log
                }, f, indent=2)
            logger.info(f"📊 Detection log saved: {log_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save detection log: {e}")


def main():
    """Enhanced main function with better error handling and configuration."""
    # Configuration
    MODEL_PATH = 'models/best.pt'
    ALARM_PATH = 'audio/alarm.wav'
    CONFIDENCE_THRESHOLD = 0.5
    USE_VIDEO_FILE = True
    VIDEO_PATH = "videos/Cam15.mp4"
    CAMERA_INDEX = 0
    OUTPUT_FOLDER = Path("D:/Autonomous Vehicle Crash Detection and Collision Risk Prediction System for Faster Emergency Response/outputs")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced Accident Detection System")
    logger.info("=" * 60)
    
    # Initialize detector
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    detector.load_alarm(ALARM_PATH)
    
    # Open video source
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        logger.info(f"📹 Using video file: {VIDEO_PATH}")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        logger.info(f"📷 Using camera index: {CAMERA_INDEX}")
    
    if not cap.isOpened():
        logger.error("❌ Failed to open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"📊 Video properties: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
    
    # Setup output
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_FOLDER / f"detection_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Black Box Recorder (CPU-friendly) 
    PRE_SEC = 5
    POST_SEC = 5
    pre_buffer = deque(maxlen=int(fps * PRE_SEC))
    recording = False
    post_frames_remaining = 0
    blackbox_dir = OUTPUT_FOLDER / "blackbox"
    blackbox_dir.mkdir(parents=True, exist_ok=True)
    jpeg_quality = 70  # trade-off quality/size for memory savings
    bb_writer = None
    bb_path = None
    # Processing metrics
    frame_count = 0
    fps_calc = deque(maxlen=30)
    
    pause_state = False
    frame_history = deque(maxlen=300)  # Store ~10 seconds at 30fps for rewind
    current_display_frame = None
    current_detections = []
    
    logger.info(f"💾 Output will be saved to: {output_path}")
    logger.info("🎬 Processing started... Press 'q' to quit")
    
    try:
        current_frame = None  # Hold the paused frame
        
        while cap.isOpened():
            # If paused, skip cap.read() to truly freeze; use stored frame
            if pause_state:
                if current_frame is None:
                    # First time pausing - read one frame then freeze
                    success, current_frame = cap.read()
                # Don't read new frames when paused
                frame = current_frame
                success = frame is not None
            else:
                # Normal operation: read next frame
                start_time = cv2.getTickCount()
                success, frame = cap.read()
            
            if not success:
                logger.info("📹 End of video stream reached")
                break

            try:
                if success:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    ret, encimg = cv2.imencode('.jpg', frame, encode_params)
                    if ret:
                        pre_buffer.append(encimg.tobytes())
            except Exception:
                # Keep the recorder best-effort and non-blocking
                pass
            
            frame_count += 1
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)

            try:
                if len(detections) > 0 and not recording:
                    timestamp_bb = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bb_path = blackbox_dir / f"blackbox_{timestamp_bb}.mp4"
                    bb_writer = cv2.VideoWriter(str(bb_path), fourcc, fps, (width, height))

                    # write pre-buffer frames (decode JPEG bytes)
                    for jpg_bytes in pre_buffer:
                        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if dec is not None:
                            bb_writer.write(dec)

                    # write current (original) frame as first post frame
                    bb_writer.write(frame)
                    recording = True
                    post_frames_remaining = int(fps * POST_SEC)
                    logger.info(f"🔴 Black box recording started: {bb_path}")
                elif recording:
                    if bb_writer is not None:
                        bb_writer.write(frame)
                    post_frames_remaining -= 1
                    if post_frames_remaining <= 0:
                        recording = False
                        if bb_writer is not None:
                            bb_writer.release()
                            bb_writer = None
                        logger.info(f"⏺️ Black box saved: {bb_path}")
            except Exception as e:
                logger.error(f"Black box recorder error: {e}")
        
            
            # Calculate FPS
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_calc.append(current_fps)
            avg_fps = np.mean(fps_calc)
            
            # Add overlay
            display_frame = detector.add_stats_overlay(annotated_frame, detections, avg_fps)
            
            current_display_frame = display_frame.copy()
            current_detections = detections.copy()
            frame_history.append((display_frame.copy(), detections.copy()))
           
            if not pause_state:
                # Only write to output and display when NOT paused
                out.write(display_frame)
                cv2.imshow("YOLO Accident Detection", display_frame)
            else:
                # Display paused frame with status indicator
                paused_frame = display_frame.copy()
                cv2.putText(paused_frame, "[PAUSED]", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("YOLO Accident Detection", paused_frame)
            
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"📊 Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
            
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                logger.info("⏹️ User requested stop")
                break
            
            elif key == ord(' '):  # SPACE: Pause/Resume
                pause_state = not pause_state
                if pause_state:
                    current_frame = frame  # Store frozen frame
                    logger.info(f"⏸️  VIDEO PAUSED (frame frozen, detection continues)")
                else:
                    logger.info(f"▶️  VIDEO RESUMED (reading from video source)")
            
            elif key == 83:  # Right Arrow key - Enter rewind mode
                if frame_history:
                    logger.info(f"⏪ Entering rewind mode ({len(frame_history)} frames available)")
                    
                    rewind_index = len(frame_history) - 1  # Start at current (newest)
                    rewind_active = True
                    
                    while rewind_active and rewind_index >= 0:
                        rewind_frame, rewind_detections = frame_history[rewind_index]
                        
                        # Display rewind frame with navigation info
                        display_rewind = rewind_frame.copy()
                        info_text = f"REWIND MODE: Frame {rewind_index + 1}/{len(frame_history)} (← → to navigate, Q to exit)"
                        cv2.putText(display_rewind, info_text, (20, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imshow("YOLO Accident Detection", display_rewind)
                        
                        # Handle rewind mode navigation
                        rewind_key = cv2.waitKey(100) & 0xFF
                        
                        if rewind_key == 81:  # Left Arrow - go to older frame
                            if rewind_index < len(frame_history) - 1:
                                rewind_index += 1
                        elif rewind_key == 83:  # Right Arrow - go to newer frame
                            if rewind_index > 0:
                                rewind_index -= 1
                        elif rewind_key == ord('q') or rewind_key == ord('Q'):
                            rewind_active = False
                            logger.info(f"📹 Exiting rewind mode at frame {rewind_index + 1}")
                   
            
            elif key == ord('s') or key == ord('S'):  # Save current frame
                if current_display_frame is not None:
                    save_path = OUTPUT_FOLDER / f"manual_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(save_path), current_display_frame)
                    logger.info(f"💾 Frame saved: {save_path}")
        
    
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        if bb_writer is not None:
            try:
                bb_writer.release()
                logger.info("Released black box writer on cleanup")
            except Exception:
                pass
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        
        # Save detection log
        detector.save_detection_log(output_path)
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("✅ Processing complete!")
        logger.info(f"📊 Total frames processed: {frame_count}")
        logger.info(f"🎯 Total detections: {detector.total_detections}")
        logger.info(f"💾 Output saved: {output_path}")
        logger.info("=" * 60)


def run(source):
    """
    Wrapper function to accept dynamic video source (webcam index or file path).
    Preserves all original main() logic without changes.
    """
    # Configuration
    MODEL_PATH = 'models/best.pt'
    ALARM_PATH = 'audio/alarm.wav'
    CONFIDENCE_THRESHOLD = 0.5
    OUTPUT_FOLDER = Path("D:/Autonomous Vehicle Crash Detection and Collision Risk Prediction System for Faster Emergency Response/outputs")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced Accident Detection System")
    logger.info("=" * 60)
    
    # Initialize detector
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    detector.load_alarm(ALARM_PATH)
    
    # Open video source (dynamic: webcam index or file path)
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        logger.info(f"📷 Using camera index: {source}")
    else:
        logger.info(f"📹 Using video file: {source}")
    
    if not cap.isOpened():
        logger.error("❌ Failed to open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"📊 Video properties: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
    
    # Setup output
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_FOLDER / f"detection_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Black Box Recorder (CPU-friendly)
    PRE_SEC = 5
    POST_SEC = 5
    pre_buffer = deque(maxlen=int(fps * PRE_SEC))
    recording = False
    post_frames_remaining = 0
    blackbox_dir = OUTPUT_FOLDER / "blackbox"
    blackbox_dir.mkdir(parents=True, exist_ok=True)
    jpeg_quality = 70
    bb_writer = None
    bb_path = None
    
    # Processing metrics
    frame_count = 0
    fps_calc = deque(maxlen=30)
    
    # Keyboard controls setup
    pause_state = False
    frame_history = deque(maxlen=300)
    current_display_frame = None
    current_detections = []
    
    logger.info(f"💾 Output will be saved to: {output_path}")
    logger.info("🎬 Processing started... Press 'q' to quit")
    
    try:
        current_frame = None
        
        while cap.isOpened():
            # Pause state handling
            if pause_state:
                if current_frame is None:
                    success, current_frame = cap.read()
                frame = current_frame
                success = frame is not None
            else:
                start_time = cv2.getTickCount()
                success, frame = cap.read()
            
            if not success:
                logger.info("📹 End of video stream reached")
                break

            # Black Box: store compressed pre-buffer frames
            try:
                if success:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    ret, encimg = cv2.imencode('.jpg', frame, encode_params)
                    if ret:
                        pre_buffer.append(encimg.tobytes())
            except Exception:
                pass
            
            frame_count += 1
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)

            # Black Box trigger/write logic
            try:
                if len(detections) > 0 and not recording:
                    timestamp_bb = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bb_path = blackbox_dir / f"blackbox_{timestamp_bb}.mp4"
                    bb_writer = cv2.VideoWriter(str(bb_path), fourcc, fps, (width, height))

                    for jpg_bytes in pre_buffer:
                        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if dec is not None:
                            bb_writer.write(dec)

                    bb_writer.write(frame)
                    recording = True
                    post_frames_remaining = int(fps * POST_SEC)
                    logger.info(f"🔴 Black box recording started: {bb_path}")
                elif recording:
                    if bb_writer is not None:
                        bb_writer.write(frame)
                    post_frames_remaining -= 1
                    if post_frames_remaining <= 0:
                        recording = False
                        if bb_writer is not None:
                            bb_writer.release()
                            bb_writer = None
                        logger.info(f"⏺️ Black box saved: {bb_path}")
            except Exception as e:
                logger.error(f"Black box recorder error: {e}")
            
            # Calculate FPS
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_calc.append(current_fps)
            avg_fps = np.mean(fps_calc)
            
            # Add overlay
            display_frame = detector.add_stats_overlay(annotated_frame, detections, avg_fps)
            
            # Store frame for rewind
            current_display_frame = display_frame.copy()
            current_detections = detections.copy()
            frame_history.append((display_frame.copy(), detections.copy()))
            
            # Conditional display & output
            if not pause_state:
                out.write(display_frame)
                cv2.imshow("YOLO Accident Detection", display_frame)
            else:
                paused_frame = display_frame.copy()
                cv2.putText(paused_frame, "[PAUSED]", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("YOLO Accident Detection", paused_frame)
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"📊 Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("⏹️ User requested stop")
                break
            
            elif key == ord(' '):
                pause_state = not pause_state
                if pause_state:
                    current_frame = frame
                    logger.info(f"⏸️  VIDEO PAUSED (frame frozen, detection continues)")
                else:
                    logger.info(f"▶️  VIDEO RESUMED (reading from video source)")
            
            elif key == 83:
                if frame_history:
                    logger.info(f"⏪ Entering rewind mode ({len(frame_history)} frames available)")
                    rewind_index = len(frame_history) - 1
                    rewind_active = True
                    
                    while rewind_active and rewind_index >= 0:
                        rewind_frame, rewind_detections = frame_history[rewind_index]
                        display_rewind = rewind_frame.copy()
                        info_text = f"REWIND MODE: Frame {rewind_index + 1}/{len(frame_history)} (← → to navigate, Q to exit)"
                        cv2.putText(display_rewind, info_text, (20, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imshow("YOLO Accident Detection", display_rewind)
                        
                        rewind_key = cv2.waitKey(100) & 0xFF
                        
                        if rewind_key == 81:
                            if rewind_index < len(frame_history) - 1:
                                rewind_index += 1
                        elif rewind_key == 83:
                            if rewind_index > 0:
                                rewind_index -= 1
                        elif rewind_key == ord('q') or rewind_key == ord('Q'):
                            rewind_active = False
                            logger.info(f"📹 Exiting rewind mode at frame {rewind_index + 1}")
            
            elif key == ord('s') or key == ord('S'):
                if current_display_frame is not None:
                    save_path = OUTPUT_FOLDER / f"manual_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(save_path), current_display_frame)
                    logger.info(f"💾 Frame saved: {save_path}")
    
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
    finally:
        cap.release()
        out.release()
        if bb_writer is not None:
            try:
                bb_writer.release()
                logger.info("Released black box writer on cleanup")
            except Exception:
                pass
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        
        detector.save_detection_log(output_path)
        
        logger.info("=" * 60)
        logger.info("✅ Processing complete!")
        logger.info(f"📊 Total frames processed: {frame_count}")
        logger.info(f"🎯 Total detections: {detector.total_detections}")
        logger.info(f"💾 Output saved: {output_path}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()