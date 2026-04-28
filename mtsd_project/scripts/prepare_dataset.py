"""
prepare_dataset.py
──────────────────
Converts MTSD annotations into YOLO format and builds the directory structure
expected by YOLOv8 training.

Output structure:
    data/
        images/
            train/  (symlinks to original MTSD images)
            val/
            test/
        labels/
            train/  (generated YOLO .txt files)
            val/
            test/

YOLO label format (one line per object):
    <class_id> <cx> <cy> <w> <h>
    All values normalized 0-1. class_id is always 0 (parking_sign).

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --no-symlinks   # copies files instead of symlinking
    python scripts/prepare_dataset.py --na-only       # only NA-confident images
"""

import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtsd"
IMAGES_DIR   = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
SPLITS_DIR   = DATASET_ROOT / "splits"
OUTPUT_DIR   = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

NA_KEYS_FILE = PROJECT_ROOT / "outputs" / "na_keys.txt"

# ── Label definitions ─────────────────────────────────────────────────────────

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

POSITIVE_LABELS = POSITIVE_LABELS_NA | POSITIVE_LABELS_NON_NA

EXCLUDE_LABELS = {"other-sign"}

CLASS_ID = 0   # single class: parking_sign


def is_usable(obj: dict) -> bool:
    props = obj.get("properties", {})
    return not any([
        props.get("ambiguous", False),
        props.get("occluded", False),
        props.get("out-of-frame", False),
        props.get("dummy", False),
    ])


def annotation_to_yolo_lines(ann: dict) -> list[str]:
    """
    Convert one MTSD annotation dict to a list of YOLO label lines.
    Returns empty list if no usable positive objects exist (hard negative image).
    """
    img_w = ann.get("width", 0)
    img_h = ann.get("height", 0)
    if img_w == 0 or img_h == 0:
        return []

    lines = []
    for obj in ann.get("objects", []):
        if not is_usable(obj):
            continue
        label = obj.get("label", "")
        if label not in POSITIVE_LABELS:
            continue

        bbox = obj.get("bbox", {})
        x1 = bbox.get("xmin", 0)
        y1 = bbox.get("ymin", 0)
        x2 = bbox.get("xmax", 0)
        y2 = bbox.get("ymax", 0)

        # Handle panorama cross-boundary boxes
        if x2 < x1:
            cross = bbox.get("cross_boundary", {})
            left  = cross.get("left", {})
            right = cross.get("right", {})
            # Process left and right halves as separate boxes
            for half in [left, right]:
                if not half:
                    continue
                hx1, hy1 = half.get("xmin", 0), half.get("ymin", 0)
                hx2, hy2 = half.get("xmax", 0), half.get("ymax", 0)
                line = _make_yolo_line(hx1, hy1, hx2, hy2, img_w, img_h)
                if line:
                    lines.append(line)
            continue

        line = _make_yolo_line(x1, y1, x2, y2, img_w, img_h)
        if line:
            lines.append(line)

    return lines


def _make_yolo_line(x1, y1, x2, y2, img_w, img_h) -> str | None:
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + w / 2) / img_w
    cy = (y1 + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    # Clamp to [0, 1] to handle edge annotations that slightly exceed image bounds
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))

    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def load_split(split_name: str) -> list[str]:
    f = SPLITS_DIR / f"{split_name}.txt"
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def load_na_keys() -> set[str] | None:
    if not NA_KEYS_FILE.exists():
        return None
    return {l.strip() for l in NA_KEYS_FILE.read_text().splitlines() if l.strip()}


def process_split(
    split: str,
    keys: list[str],
    use_symlinks: bool,
    na_keys: set[str] | None,
    stats: dict,
):
    images_out = OUTPUT_DIR / "images" / split
    labels_out = OUTPUT_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    skipped_no_image = 0
    skipped_no_ann   = 0
    skipped_na       = 0
    n_positive       = 0
    n_negative       = 0
    n_objects        = 0

    for key in keys:
        # Optional NA filter
        if na_keys is not None and key not in na_keys:
            skipped_na += 1
            continue

        img_src = IMAGES_DIR / f"{key}.jpg"
        ann_path = ANNOTATIONS_DIR / f"{key}.json"

        if not img_src.exists():
            skipped_no_image += 1
            continue
        if not ann_path.exists():
            skipped_no_ann += 1
            continue

        # Load annotation and convert to YOLO
        with open(ann_path) as f:
            ann = json.load(f)

        yolo_lines = annotation_to_yolo_lines(ann)

        # Write label file (empty = hard negative, that's valid for YOLO)
        label_dst = labels_out / f"{key}.txt"
        label_dst.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        # Link or copy image
        img_dst = images_out / f"{key}.jpg"
        if not img_dst.exists():
            if use_symlinks:
                img_dst.symlink_to(img_src.resolve())
            else:
                shutil.copy2(img_src, img_dst)

        if yolo_lines:
            n_positive += 1
            n_objects  += len(yolo_lines)
        else:
            n_negative += 1

    stats[split] = {
        "total_processed": n_positive + n_negative,
        "positive_images": n_positive,
        "negative_images": n_negative,
        "total_objects":   n_objects,
        "skipped_no_image": skipped_no_image,
        "skipped_no_ann":   skipped_no_ann,
        "skipped_na":       skipped_na,
    }


def write_yaml(na_only: bool):
    """Write the dataset YAML file that YOLOv8 training expects."""
    yaml_path = PROJECT_ROOT / "parking_signs.yaml"
    tag = " (North America only)" if na_only else " (all regions)"
    content = f"""# Parking Sign Detection Dataset{tag}
# Auto-generated by prepare_dataset.py

path: {OUTPUT_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: parking_sign
"""
    yaml_path.write_text(content)
    return yaml_path


def main(use_symlinks: bool = True, na_only: bool = False):
    na_keys = None
    if na_only:
        na_keys = load_na_keys()
        if na_keys is None:
            print("WARNING: --na-only requested but outputs/na_keys.txt not found. Processing all images.")

    print("=" * 60)
    print("MTSD → YOLO DATASET PREPARATION")
    print("=" * 60)
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"File handling     : {'symlinks' if use_symlinks else 'copies'}")
    print(f"NA filter         : {'yes' if na_keys else 'no'}")
    print()

    stats = {}
    for split in ["train", "val", "test"]:
        keys = load_split(split)
        if not keys:
            print(f"  WARNING: No keys found for split '{split}', skipping.")
            continue
        print(f"Processing {split} ({len(keys)} keys)...")
        process_split(split, keys, use_symlinks, na_keys, stats)

    yaml_path = write_yaml(na_only)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split, s in stats.items():
        total = s["positive_images"] + s["negative_images"]
        if total == 0:
            continue
        ratio = s["negative_images"] / max(s["positive_images"], 1)
        print(f"\n  {split.upper()}")
        print(f"    Total images processed : {s['total_processed']}")
        print(f"    Positive images        : {s['positive_images']}  ({100*s['positive_images']/max(total,1):.1f}%)")
        print(f"    Negative images        : {s['negative_images']}  ({100*s['negative_images']/max(total,1):.1f}%)")
        print(f"    Total sign objects     : {s['total_objects']}")
        print(f"    Neg:Pos ratio          : {ratio:.1f}:1")
        if s["skipped_no_image"]:
            print(f"    Skipped (no image)     : {s['skipped_no_image']}")
        if s["skipped_no_ann"]:
            print(f"    Skipped (no annotation): {s['skipped_no_ann']}")
        if s["skipped_na"]:
            print(f"    Skipped (outside NA)   : {s['skipped_na']}")

    print(f"\n  Dataset YAML written   : {yaml_path}")
    print("\nDone. Ready for training.")
    print(f"  Next: python scripts/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Copy image files instead of symlinking (slower, uses more disk)")
    parser.add_argument("--na-only", action="store_true",
                        help="Only include NA-confident images (requires outputs/na_keys.txt)")
    args = parser.parse_args()
    main(use_symlinks=not args.no_symlinks, na_only=args.na_only)