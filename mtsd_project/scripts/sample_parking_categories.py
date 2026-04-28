"""
sample_parking_categories.py
----------------------------
Pick one usable example image for each parking-related label/category used in training,
then save:
  1. a CSV summary
  2. a collage of full images with the chosen bounding box drawn
  3. a collage of cropped sign regions

Usage:
    python scripts/sample_parking_categories.py
    python scripts/sample_parking_categories.py --seed 42
    python scripts/sample_parking_categories.py --split train
    python scripts/sample_parking_categories.py --seed 7 --split val

Outputs:
    outputs/category_samples/
        category_samples_<split>_seed<seed>.csv
        category_full_collage_<split>_seed<seed>.jpg
        category_crop_collage_<split>_seed<seed>.jpg
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtsd"
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
SPLITS_DIR = DATASET_ROOT / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "category_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parking labels used in training
POSITIVE_LABELS_NA = {
    "regulatory--no-parking--g1",
    "regulatory--no-stopping--g15",
    "regulatory--no-stopping--g8",
    "regulatory--no-parking-or-no-stopping--g1",
    "regulatory--no-parking-or-no-stopping--g3",
    "regulatory--end-of-no-parking--g1",
    "complementary--tow-away-zone--g1",
    "information--parking--g1",
    "information--parking--g3",
    "information--parking--g5",
    "information--parking--g45",
    "regulatory--parking-restrictions--g2",
}

POSITIVE_LABELS_NON_NA = {
    "regulatory--no-parking--g2",
    "regulatory--no-parking--g5",
    "regulatory--no-stopping--g2",
    "regulatory--no-stopping--g4",
    "regulatory--no-stopping--g5",
    "regulatory--no-parking-or-no-stopping--g2",
    "information--no-parking--g3",
    "information--parking--g2",
    "information--parking--g6",
}

POSITIVE_LABELS = sorted(POSITIVE_LABELS_NA | POSITIVE_LABELS_NON_NA)


def is_usable(obj):
    props = obj.get("properties", {})
    return not any([
        props.get("ambiguous", False),
        props.get("occluded", False),
        props.get("out-of-frame", False),
        props.get("dummy", False),
    ])


def load_split_keys(split_name):
    split_file = SPLITS_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def safe_crop_box(x1, y1, x2, y2, w, h, pad_frac=0.12):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return cx1, cy1, cx2, cy2


def fit_with_padding(img, size=(320, 240), bg=(245, 245, 245)):
    canvas = Image.new("RGB", size, bg)
    img = img.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def fit_crop_square(img, size=(220, 220), bg=(245, 245, 245)):
    canvas = Image.new("RGB", size, bg)
    img = img.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def get_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def shorten_label(label, max_len=28):
    if len(label) <= max_len:
        return label
    return label[:max_len - 3] + "..."


def make_full_image_tile(image_path, bbox, label, tile_size=(320, 280)):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)

    fitted = fit_with_padding(img, size=(tile_size[0], tile_size[1] - 40))
    tile = Image.new("RGB", tile_size, (255, 255, 255))
    tile.paste(fitted, (0, 0))

    draw_tile = ImageDraw.Draw(tile)
    font = get_font(15)
    text = shorten_label(label, 36)
    draw_tile.text((8, tile_size[1] - 32), text, fill=(0, 0, 0), font=font)

    return tile


def make_crop_tile(image_path, bbox, label, tile_size=(220, 260)):
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = safe_crop_box(x1, y1, x2, y2, img.width, img.height)
    crop = img.crop((cx1, cy1, cx2, cy2))

    fitted = fit_crop_square(crop, size=(tile_size[0], tile_size[1] - 40))
    tile = Image.new("RGB", tile_size, (255, 255, 255))
    tile.paste(fitted, (0, 0))

    draw_tile = ImageDraw.Draw(tile)
    font = get_font(15)
    text = shorten_label(label, 28)
    draw_tile.text((8, tile_size[1] - 32), text, fill=(0, 0, 0), font=font)

    return tile


def build_collage(tiles, tile_size, cols, out_path, title=None):
    if not tiles:
        raise ValueError("No tiles to place in collage.")

    rows = math.ceil(len(tiles) / cols)
    title_h = 50 if title else 0
    canvas_w = cols * tile_size[0]
    canvas_h = rows * tile_size[1] + title_h

    collage = Image.new("RGB", (canvas_w, canvas_h), (235, 235, 235))
    draw = ImageDraw.Draw(collage)

    if title:
        font = get_font(24)
        draw.text((12, 12), title, fill=(0, 0, 0), font=font)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * tile_size[0]
        y = title_h + r * tile_size[1]
        collage.paste(tile, (x, y))

    collage.save(out_path, quality=95)
    print(f"Saved collage -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Choose examples only from this split")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_keys = set(load_split_keys(args.split))

    # Gather all usable candidate objects per label
    candidates = {label: [] for label in POSITIVE_LABELS}

    ann_paths = sorted(ANNOTATIONS_DIR.glob("*.json"))
    for ann_path in ann_paths:
        key = ann_path.stem
        if key not in split_keys:
            continue

        img_path = IMAGES_DIR / f"{key}.jpg"
        if not img_path.exists():
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        for obj in ann.get("objects", []):
            label = obj.get("label", "")
            if label not in candidates:
                continue
            if not is_usable(obj):
                continue

            bbox = obj.get("bbox", {})
            x1 = int(round(bbox.get("xmin", 0)))
            y1 = int(round(bbox.get("ymin", 0)))
            x2 = int(round(bbox.get("xmax", 0)))
            y2 = int(round(bbox.get("ymax", 0)))

            # skip weird invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            candidates[label].append({
                "image_key": key,
                "image_path": str(img_path),
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
            })

    selected = []
    missing = []

    for label in POSITIVE_LABELS:
        pool = candidates[label]
        if not pool:
            missing.append(label)
            continue
        chosen = rng.choice(pool)
        chosen["label"] = label
        selected.append(chosen)

    print(f"Selected {len(selected)} categories from split='{args.split}' with seed={args.seed}")
    if missing:
        print("No examples found for:")
        for m in missing:
            print("  -", m)

    # Save CSV
    csv_path = OUTPUT_DIR / f"category_samples_{args.split}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "image_key", "image_path", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "width", "height", "area"]
        )
        writer.writeheader()
        for row in selected:
            x1, y1, x2, y2 = row["bbox"]
            writer.writerow({
                "label": row["label"],
                "image_key": row["image_key"],
                "image_path": row["image_path"],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "width": row["width"],
                "height": row["height"],
                "area": row["area"],
            })
    print(f"Saved CSV -> {csv_path}")

    # Build tiles
    full_tiles = []
    crop_tiles = []

    for row in selected:
        full_tiles.append(
            make_full_image_tile(
                image_path=row["image_path"],
                bbox=row["bbox"],
                label=row["label"],
                tile_size=(320, 280),
            )
        )
        crop_tiles.append(
            make_crop_tile(
                image_path=row["image_path"],
                bbox=row["bbox"],
                label=row["label"],
                tile_size=(220, 260),
            )
        )

    # Save collages
    full_collage_path = OUTPUT_DIR / f"category_full_collage_{args.split}_seed{args.seed}.jpg"
    crop_collage_path = OUTPUT_DIR / f"category_crop_collage_{args.split}_seed{args.seed}.jpg"

    build_collage(
        full_tiles,
        tile_size=(320, 280),
        cols=7,
        out_path=full_collage_path,
        # title=f"One sample per parking category | split={args.split} | seed={args.seed}",
    )

    build_collage(
        crop_tiles,
        tile_size=(220, 260),
        cols=7,
        out_path=crop_collage_path,
        # title=f"Cropped sign samples per parking category | split={args.split} | seed={args.seed}",
    )

    print("\nDone.")
    print("Tip: change --seed to get a different but reproducible set of examples.")


if __name__ == "__main__":
    main()