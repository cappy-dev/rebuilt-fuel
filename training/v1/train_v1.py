import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from ultralytics import YOLO

SOURCE_DIR = Path("./dataset1/annotated")  # folder with train.txt, images, labels
DEST_DIR = Path("./dataset_split")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MODEL_NAME = "yolo26n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
WORKERS = 4

# 1. Read class names
with open(SOURCE_DIR / "obj.names", "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]
num_classes = len(class_names)
print(f"Classes: {class_names}")

# 2. Read image paths from train.txt
with open(SOURCE_DIR / "train.txt", "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

# Ensure paths are absolute or relative to SOURCE_DIR
image_paths = [SOURCE_DIR / "obj_train_data" / Path(p).name for p in image_paths]
valid_paths = [p for p in image_paths if p.exists()]
print(f"Total images found: {len(valid_paths)}")

# 3. Split into train/val
train_paths, val_paths = train_test_split(
    valid_paths,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_SEED,
)

# 4. Create folder structure
for split in ["train", "val"]:
    os.makedirs(DEST_DIR / split / "images", exist_ok=True)
    os.makedirs(DEST_DIR / split / "labels", exist_ok=True)

def copy_pair(img_path, split):
    # Copy image
    dest_img = DEST_DIR / split / "images" / img_path.name
    shutil.copy2(img_path, dest_img)
    # Copy corresponding label
    label_name = img_path.stem + ".txt"
    src_label = img_path.parent / label_name
    if src_label.exists():
        dest_label = DEST_DIR / split / "labels" / label_name
        shutil.copy2(src_label, dest_label)
    else:
        print(f"Warning: label missing for {img_path.name}")

print("Copying train set...")
for p in train_paths:
    copy_pair(p, "train")

print("Copying val set...")
for p in val_paths:
    copy_pair(p, "val")

# 5. Create data.yaml
data_yaml = DEST_DIR / "data.yaml"
data_yaml_content = f"""path: {DEST_DIR.resolve()}  # absolute path
train: train/images
val: val/images
nc: {num_classes}
names: {class_names}
"""
with open(data_yaml, "w") as f:
    f.write(data_yaml_content)
print(f"data.yaml written to {data_yaml}")

# 6. Start YOLO26 Training – output to training/v1/
model = YOLO(MODEL_NAME)
results = model.train(
    data=str(data_yaml),
    epochs=300,               # longer training
    patience=30,              # stop early if no improvement for 30 epochs
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=WORKERS,
    device="cpu",
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    project="training",       # top-level folder in the repo
    name="v1",                # subfolder for this experiment
    exist_ok=True,            # overwrite if v1 already exists
)
print("Training complete. Best model saved at:", results.save_dir)
