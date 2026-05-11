import os
import shutil
import random

# ====== CHANGE ONLY THIS IF YOUR PATH IS DIFFERENT ======
BASE_DIR = r"E:\vision_ads\dataset"

IMAGES_DIR = os.path.join(BASE_DIR, "image")
LABELS_DIR = os.path.join(BASE_DIR, "label")

OUTPUT_DIR = r"E:\vision_ads\data"
# =======================================================

# Output folders
PATHS = {
    "train_img": os.path.join(OUTPUT_DIR, "train", "images"),
    "train_lbl": os.path.join(OUTPUT_DIR, "train", "labels"),
    "val_img":   os.path.join(OUTPUT_DIR, "valid", "images"),
    "val_lbl":   os.path.join(OUTPUT_DIR, "valid", "labels"),
    "test_img":  os.path.join(OUTPUT_DIR, "test", "images"),
    "test_lbl":  os.path.join(OUTPUT_DIR, "test", "labels"),
}

# Create folders
for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

# Check paths
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

if not os.path.exists(LABELS_DIR):
    raise FileNotFoundError(f"Labels folder not found: {LABELS_DIR}")

# Get all images
images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
random.shuffle(images)

total = len(images)

train_end = int(0.7 * total)
val_end = int(0.9 * total)

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

def copy_files(files, img_src, lbl_src, img_dst, lbl_dst):
    for f in files:
        shutil.copy(os.path.join(img_src, f), os.path.join(img_dst, f))
        txt = f.replace(".jpg", ".txt")
        shutil.copy(os.path.join(lbl_src, txt), os.path.join(lbl_dst, txt))

copy_files(train_files, IMAGES_DIR, LABELS_DIR, PATHS["train_img"], PATHS["train_lbl"])
copy_files(val_files, IMAGES_DIR, LABELS_DIR, PATHS["val_img"], PATHS["val_lbl"])
copy_files(test_files, IMAGES_DIR, LABELS_DIR, PATHS["test_img"], PATHS["test_lbl"])

print("✅ SPLIT COMPLETED SUCCESSFULLY")
print(f"Total images : {total}")
print(f"Train images : {len(train_files)}")
print(f"Val images   : {len(val_files)}")
print(f"Test images  : {len(test_files)}")
