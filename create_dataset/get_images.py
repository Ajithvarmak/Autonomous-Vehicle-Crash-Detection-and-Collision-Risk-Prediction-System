import cv2
import os

video = cv2.VideoCapture("frames/Cam22.mp4")

# Main dataset folder
base_dir = "dataset5"

# Subfolders
img_folder = os.path.join(base_dir, "images21")
label_folder = os.path.join(base_dir, "labels21")

os.makedirs(img_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

count = 0

while True:
    ok, frame = video.read()
    if not ok:
        break

    if count % 10 == 0:
        img_name = f"frame22_{count}.jpg"
        txt_name = f"frame22_{count}.txt"

        cv2.imwrite(os.path.join(img_folder, img_name), frame)
        open(os.path.join(label_folder, txt_name), "w").close()

    count += 1

video.release()
print("✅ Images and TXT files saved inside dataset folder")
