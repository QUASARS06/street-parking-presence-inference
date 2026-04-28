"""
explore_mtsd.py
───────────────
Exploratory analysis of the MTSD dataset with:
  - Property-based annotation filtering (ambiguous, occluded, out-of-frame, dummy)
  - Binary label mapping (positive / negative / exclude)
  - Optional North America filter (pass --na-only after running fetch_na_keys.py)

Usage:
    python scripts/explore_mtsd.py                  # all images
    python scripts/explore_mtsd.py --na-only        # North America only
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtsd"
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
SPLITS_DIR = DATASET_ROOT / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NA_KEYS_FILE = OUTPUT_DIR / "na_keys.txt"

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_LABELS = {
    # ── Permissive / information parking ─────────────────────────────────────
    "information--parking--g1",
    "information--parking--g2",
    "information--parking--g3",
    "information--parking--g5",
    "information--parking--g6",
    "information--parking--g45",
    "information--no-parking--g3",

    # ── No parking ────────────────────────────────────────────────────────────
    "regulatory--no-parking--g1",
    "regulatory--no-parking--g2",
    "regulatory--no-parking--g5",
    "regulatory--end-of-no-parking--g1",

    # ── No stopping / no standing ─────────────────────────────────────────────
    "regulatory--no-stopping--g2",
    "regulatory--no-stopping--g4",
    "regulatory--no-stopping--g5",
    "regulatory--no-stopping--g8",
    "regulatory--no-stopping--g15",

    # ── Combined no-parking / no-stopping ────────────────────────────────────
    "regulatory--no-parking-or-no-stopping--g1",
    "regulatory--no-parking-or-no-stopping--g2",
    "regulatory--no-parking-or-no-stopping--g3",

    # ── Parking restrictions ──────────────────────────────────────────────────
    "regulatory--parking-restrictions--g2",

    # ── Tow-away ──────────────────────────────────────────────────────────────
    "complementary--tow-away-zone--g1",
}

EXCLUDE_LABELS = {
    "other-sign",   # catch-all with no consistent visual identity
}


def is_usable(obj: dict) -> bool:
    props = obj.get("properties", {})
    return not any([
        props.get("ambiguous", False),
        props.get("occluded", False),
        props.get("out-of-frame", False),
        props.get("dummy", False),
    ])


def classify_label(label: str) -> str:
    if label in POSITIVE_LABELS:
        return "positive"
    if label in EXCLUDE_LABELS:
        return "exclude"
    return "negative"


def load_split(split_name: str) -> list:
    split_file = SPLITS_DIR / f"{split_name}.txt"
    if not split_file.exists():
        return []
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_na_keys():
    if not NA_KEYS_FILE.exists():
        print(f"  WARNING: NA keys file not found at {NA_KEYS_FILE}")
        print("     Run fetch_na_keys.py first, then rerun with --na-only.")
        return None
    with open(NA_KEYS_FILE) as f:
        keys = {line.strip() for line in f if line.strip()}
    print(f"  Loaded {len(keys)} North America keys from {NA_KEYS_FILE}")
    return keys


def main(na_only=False):
    na_keys = None
    if na_only:
        na_keys = load_na_keys()
        if na_keys is None:
            return

    all_annotation_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    if na_only and na_keys is not None:
        annotation_files = [p for p in all_annotation_files if p.stem in na_keys]
    else:
        annotation_files = all_annotation_files

    image_files = list(IMAGES_DIR.glob("*.jpg"))
    geo_tag = " (North America only)" if na_only else " (all regions)"

    print("=" * 65)
    print(f"MTSD DATASET SUMMARY{geo_tag}")
    print("=" * 65)
    print(f"Dataset root         : {DATASET_ROOT}")
    print(f"Total images on disk : {len(image_files)}")
    print(f"Annotations analyzed : {len(annotation_files)}")
    if na_only:
        print(f"  (filtered from {len(all_annotation_files)} total annotations)")

    for split_name in ["train", "val", "test"]:
        split_items = load_split(split_name)
        if na_only and na_keys:
            split_items = [k for k in split_items if k in na_keys]
        print(f"  {split_name:>5} split        : {len(split_items)}")

    # ── Per-annotation pass ───────────────────────────────────────────────────
    raw_class_counter = Counter()
    clean_class_counter = Counter()
    objects_per_image_raw = []
    objects_per_image_clean = []
    bbox_widths, bbox_heights, bbox_areas, relative_areas = [], [], [], []

    images_with_positive = 0
    images_with_no_positive = 0
    images_excluded_only = 0
    missing_images = 0
    pano_count = 0

    for ann_path in annotation_files:
        image_path = IMAGES_DIR / f"{ann_path.stem}.jpg"
        if not image_path.exists():
            missing_images += 1

        with open(ann_path) as f:
            data = json.load(f)

        if data.get("ispano", False):
            pano_count += 1

        img_w = data.get("width", 0)
        img_h = data.get("height", 0)
        img_area = img_w * img_h if img_w and img_h else 0
        objects = data.get("objects", [])
        objects_per_image_raw.append(len(objects))

        usable = [o for o in objects if is_usable(o)]
        objects_per_image_clean.append(len(usable))

        for obj in objects:
            raw_class_counter[obj.get("label", "UNKNOWN")] += 1

        has_positive = False
        has_any_usable = False

        for obj in usable:
            label = obj.get("label", "UNKNOWN")
            bin_ = classify_label(label)
            if bin_ == "exclude":
                continue

            clean_class_counter[label] += 1
            has_any_usable = True
            if bin_ == "positive":
                has_positive = True

            bbox = obj.get("bbox", {})
            x1, y1 = bbox.get("xmin", 0), bbox.get("ymin", 0)
            x2, y2 = bbox.get("xmax", 0), bbox.get("ymax", 0)
            if x2 >= x1:
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    bbox_widths.append(w)
                    bbox_heights.append(h)
                    area = w * h
                    bbox_areas.append(area)
                    if img_area > 0:
                        relative_areas.append(area / img_area)

        if has_positive:
            images_with_positive += 1
        elif has_any_usable:
            images_with_no_positive += 1
        else:
            images_excluded_only += 1

    # ── OBJECT STATS ──────────────────────────────────────────────────────────
    total_raw = sum(raw_class_counter.values())
    total_clean = sum(clean_class_counter.values())

    print("\n" + "=" * 65)
    print("OBJECT STATS")
    print("=" * 65)
    print(f"Total raw labeled objects     : {total_raw}")
    print(f"Total clean objects (filtered): {total_clean}")
    print(f"  Dropped by property filter  : {total_raw - total_clean}")
    print(f"Number of raw classes         : {len(raw_class_counter)}")
    print(f"Missing image files           : {missing_images}")
    print(f"Panorama images               : {pano_count}")

    if objects_per_image_raw:
        print(f"\nObjects per image (raw)   -- mean: {statistics.mean(objects_per_image_raw):.2f}"
              f"  median: {statistics.median(objects_per_image_raw):.2f}"
              f"  max: {max(objects_per_image_raw)}")
    if objects_per_image_clean:
        print(f"Objects per image (clean) -- mean: {statistics.mean(objects_per_image_clean):.2f}"
              f"  median: {statistics.median(objects_per_image_clean):.2f}"
              f"  max: {max(objects_per_image_clean)}")

    # ── BBOX STATS ────────────────────────────────────────────────────────────
    if bbox_widths:
        print("\n" + "=" * 65)
        print("BOUNDING BOX STATS  (clean objects only)")
        print("=" * 65)
        print(f"Avg bbox width         : {statistics.mean(bbox_widths):.1f} px")
        print(f"Avg bbox height        : {statistics.mean(bbox_heights):.1f} px")
        print(f"Avg bbox area          : {statistics.mean(bbox_areas):.0f} px2")
        print(f"Median bbox area       : {statistics.median(bbox_areas):.0f} px2")
        if relative_areas:
            print(f"Avg relative area      : {statistics.mean(relative_areas)*100:.3f}% of image")
            print(f"Median relative area   : {statistics.median(relative_areas)*100:.3f}% of image")
            tiny = sum(1 for r in relative_areas if r < 0.001)
            print(f"Tiny signs (<0.1% img) : {tiny} ({100*tiny/len(relative_areas):.1f}%)")

    # ── IMAGE-LEVEL BINARY SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("IMAGE-LEVEL BINARY LABEL SUMMARY  (after property filter)")
    print("=" * 65)
    total_imgs = len(annotation_files)
    print(f"Total annotation files         : {total_imgs}")
    print(f"  POSITIVE (>=1 parking sign)  : {images_with_positive}"
          f"  ({100*images_with_positive/total_imgs:.1f}%)")
    print(f"  NEGATIVE (no parking signs)  : {images_with_no_positive}"
          f"  ({100*images_with_no_positive/total_imgs:.1f}%)")
    print(f"  EXCLUDED (only junk labels)  : {images_excluded_only}"
          f"  ({100*images_excluded_only/total_imgs:.1f}%)")
    ratio = images_with_no_positive / max(images_with_positive, 1)
    print(f"\n  Neg:Pos ratio                : {ratio:.1f}:1")
    if ratio > 10:
        print("  WARNING: Heavy class imbalance -- use weighted loss or oversample positives.")

    # ── POSITIVE CLASS BREAKDOWN ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("POSITIVE CLASS COUNTS  (clean, usable instances)")
    print("=" * 65)
    positive_counts = {
        lbl: clean_class_counter[lbl]
        for lbl in POSITIVE_LABELS
        if clean_class_counter[lbl] > 0
    }
    for label, count in sorted(positive_counts.items(), key=lambda x: -x[1]):
        bar = "X" * min(40, count // 20)
        print(f"  {label:50s} {count:5d}  {bar}")

    total_pos_instances = sum(positive_counts.values())
    print(f"\n  Total positive instances     : {total_pos_instances}")

    missing_pos = {lbl for lbl in POSITIVE_LABELS if clean_class_counter[lbl] == 0}
    if missing_pos:
        print(f"\n  Labels in map but 0 instances in this subset:")
        for lbl in sorted(missing_pos):
            print(f"    {lbl}")

    # ── TOP 50 RAW CLASSES ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TOP 50 RAW CLASSES")
    print("=" * 65)
    for label, count in raw_class_counter.most_common(50):
        bin_ = classify_label(label)
        tag = " <- POSITIVE" if bin_ == "positive" else (
              " <- EXCLUDE"  if bin_ == "exclude"  else "")
        print(f"  {label:50s} {count:6d}{tag}")

    # ── WRITE OUTPUTS ─────────────────────────────────────────────────────────
    suffix = "_na" if na_only else ""

    all_labels_path = OUTPUT_DIR / f"all_labels{suffix}.txt"
    with open(all_labels_path, "w") as f:
        for label, count in raw_class_counter.most_common():
            f.write(f"{label}\t{count}\t{classify_label(label)}\n")
    print(f"\nSaved label list        -> {all_labels_path}")

    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({
            "positive": sorted(POSITIVE_LABELS),
            "exclude": sorted(EXCLUDE_LABELS),
            "note": (
                "negative = any image with zero positive-class objects after "
                "property and exclude filters. Excluded images should be dropped."
            )
        }, f, indent=2)
    print(f"Saved label map         -> {label_map_path}")

    csv_path = OUTPUT_DIR / f"image_binary_labels{suffix}.csv"
    with open(csv_path, "w") as f:
        f.write("image_key,binary_label,n_positive_objects,n_total_clean_objects\n")
        for ann_path in annotation_files:
            with open(ann_path) as af:
                data = json.load(af)
            usable = [o for o in data.get("objects", []) if is_usable(o)]
            n_pos = sum(1 for o in usable if classify_label(o.get("label", "")) == "positive")
            n_clean = sum(1 for o in usable if classify_label(o.get("label", "")) != "exclude")
            f.write(f"{ann_path.stem},{1 if n_pos > 0 else 0},{n_pos},{n_clean}\n")
    print(f"Saved binary labels CSV -> {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--na-only",
        action="store_true",
        help="Filter to North America keys only (requires outputs/na_keys.txt from fetch_na_keys.py)"
    )
    args = parser.parse_args()
    main(na_only=args.na_only)