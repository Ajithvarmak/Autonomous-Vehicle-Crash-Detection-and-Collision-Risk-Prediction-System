# system_main.py

import os


def main():

    print("====================================")
    print("  VEHICLE SAFETY MONITORING SYSTEM  ")
    print("====================================")
    print("1️⃣  Accident Detection")
    print("2️⃣  Accident Prediction")
    print("------------------------------------")

    mode_choice = input("Select Mode (1 or 2): ").strip()

    if mode_choice not in ["1", "2"]:
        print("Invalid selection.")
        return

    print("\nSelect Video Source:")
    print("1️⃣  Webcam")
    print("2️⃣  Video File")
    source_choice = input("Select Source (1 or 2): ").strip()

    if source_choice == "1":
        source = 0

    elif source_choice == "2":
        video_name = input("Enter video file name (example: video2.mp4): ").strip()
        source = os.path.join("videos", video_name)

        if not os.path.exists(source):
            print(f"Video not found: {source}")
            return
    else:
        print("Invalid source selection.")
        return

    # Run selected system
    if mode_choice == "1":
        print("\n🚨 Running Accident Detection...\n")
        from accident_detection import run
        run(source)

    elif mode_choice == "2":
        print("\n🟡 Running Accident Prediction...\n")
        from accident_prevention import run
        run(source)


if __name__ == "__main__":
    main()