from twilio.rest import Client
import logging
import requests

logger = logging.getLogger(__name__)

def get_live_location():
    try:
        data = requests.get("https://ipinfo.io/json", timeout=5).json()
        
        location = f"{data.get('city')}, {data.get('region')}, {data.get('country')}"
        coords = data.get("loc", "")
        maps_link = f"https://maps.google.com/?q={coords}" if coords else ""
        
        return location, maps_link
    except:
        return "Unknown Location", ""


def send_sms_alert(severity="CRITICAL"):
    """
    Send SMS alert using Twilio API.
    Optimized for Trial account - short message within 160 character limit.
    """

    account_sid = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    auth_token = "your_auth_token"
    from_number = "purchased Twilio number"
    to_number = "reveiver's phone number"

    # 📍 GET LOCATION (ADDED)
    location, maps_link = get_live_location()

    # ✅ Message with location
    message_body = (
        f"{severity} accident!\n"
        f"{location}\n"
        f"{maps_link}"
    )

    msg_length = len(message_body)

    if msg_length > 160:
        message_body = message_body[:157] + "..."
        logger.warning("Message truncated to fit 160 char limit")

    try:
        client = Client(account_sid, auth_token)

        print(f"Sending SMS ({msg_length} chars) to {to_number}...")
        logger.info(f"Attempting SMS: {from_number} -> {to_number}")

        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )

        print(f"SMS sent! SID: {message.sid} | Status: {message.status}")
        logger.info(f"SMS sent - SID: {message.sid}")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"SMS failed: {error_msg}")
        logger.error(f"SMS sending failed: {error_msg}")

        if "21608" in error_msg:
            print("Trial account: Verify the receiver number in Twilio.")

        return False
