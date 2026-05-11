import requests

def get_live_location():
    try:
        print("Fetching location...")

        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()

        print("Full API response:")
        print(data)

        city = data.get("city", "Unknown City")
        region = data.get("region", "")
        country = data.get("country", "")
        coords = data.get("loc", "")

        location = f"{city}, {region}, {country}"

        if coords:
            maps_link = f"https://maps.google.com/?q={coords}"
        else:
            maps_link = "N/A"

        return location, maps_link

    except Exception as e:
        print("Error:", e)
        return "Unknown Location", "N/A"


# TEST
location, maplink = get_live_location()

print("\nRESULT:")
print("Location:", location)
print("Google Maps:", maplink)
