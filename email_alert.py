"""
Simple Email Alert with GPS - Minimal Code
"""

from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import cv2
import os
import logging

logger = logging.getLogger(__name__)

def get_location():
    """Get GPS location, coordinates and Google Maps link"""
    try:
        data = requests.get("https://ipinfo.io/json", timeout=5).json()
        city = data.get('city', 'Unknown')
        region = data.get('region', '')
        country = data.get('country', '')
        loc = data.get('loc', '')  # format: "lat,lon"
        # human readable location
        location = f"{city}, {region}, {country}".strip(', ')
        # Use Google Maps search query so the link opens properly
        maps_link = f"https://www.google.com/maps/search/?api=1&query={loc}" if loc else "N/A"
        return location, loc, maps_link
    except Exception:
        return "Unknown Location", "", "N/A"

def save_image(frame):
    """Save accident image"""
    filename = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ Image saved: {filename}")
    return filename

def send_alert(image_path, severity="CRITICAL"):
    """Send email with GPS location (plain text + HTML)"""
    # get location, coordinates and maps url
    location, coords, maps = get_location()

    sender = "sender email id"
    receiver = "receiver email id"
    password = "app password"

    subject = f"🚨 {severity} Accident Alert - {location if location else 'Location Unknown'}"

    # Plain text fallback
    plain = f"""
ACCIDENT DETECTED - {severity}

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Location: {location}
Coordinates: {coords if coords else 'N/A'}
Google Maps: {maps}

Evidence attached.
"""

    # HTML body with clickable Maps link
    maps_html = f'<a href="{maps}">Open in Google Maps</a>' if maps and maps != 'N/A' else 'N/A'
    html = f"""<html>
  <body>
    <h2>🚨 ACCIDENT DETECTED - {severity}</h2>
    <p><b>Time:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><b>Location:</b> {location}</p>
    <p><b>Coordinates:</b> {coords if coords else 'N/A'}</p>
    <p><b>Google Maps:</b> {maps_html}</p>
    <p>Evidence attached.</p>
  </body>
</html>"""

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject

    # Attach both plain and HTML versions (explicit UTF-8 charset)
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    # Attach image
    with open(image_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

    # Basic validation: ensure image exists before sending
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    # Send
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent: {severity} | {location} | {coords if coords else 'N/A'}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# Backwards-compatible wrapper functions (fixes import errors in the notebook)

def save_accident_image(frame):
    """Deprecated compatibility wrapper for older notebook code."""
    return save_image(frame)


def send_email_alert(image_path_or_frame, severity="CRITICAL"):
    """Deprecated compatibility wrapper. Accepts either a file path or a frame (numpy ndarray)."""
    # If passed a numpy array/frame, save it first
    try:
        import numpy as _np
        if hasattr(image_path_or_frame, "shape") and isinstance(image_path_or_frame, _np.ndarray):
            path = save_image(image_path_or_frame)
        else:
            path = image_path_or_frame
    except Exception:
        path = image_path_or_frame

    return send_alert(path, severity)