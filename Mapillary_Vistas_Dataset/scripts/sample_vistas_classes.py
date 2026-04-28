import os
import json
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DATA_ROOT = "/Users/chiragjain/Downloads/Mapillary_Vistas_Dataset"  # change this
SPLIT = "training"  # or "validation"

POLY_DIR = os.path.join(DATA_ROOT, SPLIT, "v2.0", "polygons")
IMG_DIR = os.path.join(DATA_ROOT, SPLIT, "images")

OUTPUT_DIR = "sample_class_image"

SAMPLES_PER_CLASS = 10  # number of images per class
SEED = None  # set to int for reproducibility

TARGET_CLASSES = [
    "object--support--pole",
    "object--traffic-sign--front",
    "construction--barrier--curb",
    "object--support--utility-pole",
    "object--sign--advertisement",
    "object--traffic-sign--back",
    "object--sign--store",
    "object--traffic-sign--direction-front",
    "construction--flat--curb-cut",
    "object--sign--ambiguous",
    "object--traffic-sign--direction-back",
    "object--support--pole-group",
    "object--support--traffic-sign-frame",
    "object--traffic-sign--information-parking",
    "construction--flat--parking",
    "object--traffic-sign--ambiguous",
    "object--sign--information",
    "object--sign--back",
    "object--sign--other",
    "object--traffic-sign--temporary-front",
    "object--parking-meter",
    "construction--flat--parking-aisle",
    "object--traffic-sign--temporary-back",
]

# =========================
# SETUP
# =========================

if SEED is not None:
    random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1: COLLECT IMAGE IDS PER CLASS
# =========================

class_to_images = defaultdict(list)

print("Scanning dataset...")

for fname in tqdm(os.listdir(POLY_DIR)):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(POLY_DIR, fname)

    with open(path, "r") as f:
        data = json.load(f)

    labels_in_image = set()

    for obj in data["objects"]:
        label = obj["label"]

        if label in TARGET_CLASSES:
            labels_in_image.add(label)

    for label in labels_in_image:
        class_to_images[label].append(fname.replace(".json", ""))

# =========================
# STEP 2: SAMPLE IMAGES
# =========================

sampled = {}

for cls in TARGET_CLASSES:
    imgs = class_to_images.get(cls, [])

    if len(imgs) == 0:
        print(f"[WARNING] No images found for class: {cls}")
        continue

    k = min(SAMPLES_PER_CLASS, len(imgs))
    sampled[cls] = random.sample(imgs, k)

# =========================
# DRAW FUNCTION
# =========================

def draw_annotations(img, objects, target_label):
    for obj in objects:
        if obj["label"] != target_label:
            continue

        poly = np.array(obj["polygon"], dtype=np.int32)

        # draw polygon
        cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

        # bounding box
        x_min = np.min(poly[:, 0])
        y_min = np.min(poly[:, 1])
        x_max = np.max(poly[:, 0])
        y_max = np.max(poly[:, 1])

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # label
        cv2.putText(
            img,
            target_label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return img

# =========================
# STEP 3: COPY + DRAW
# =========================

print("Generating samples...")

for cls, img_ids in sampled.items():
    safe_cls = cls.replace("/", "_")

    cls_dir = os.path.join(OUTPUT_DIR, safe_cls)
    os.makedirs(cls_dir, exist_ok=True)

    for img_id in img_ids:
        img_path = os.path.join(IMG_DIR, img_id + ".jpg")
        poly_path = os.path.join(POLY_DIR, img_id + ".json")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)

        with open(poly_path, "r") as f:
            data = json.load(f)

        img = draw_annotations(img, data["objects"], cls)

        out_path = os.path.join(cls_dir, f"{img_id}.jpg")
        cv2.imwrite(out_path, img)

print("Done. Samples saved to:", OUTPUT_DIR)