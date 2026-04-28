#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict


STRONG_CURB_COLORS = {"red", "yellow", "green", "white"}


DEFAULT_TYPE_COUNTS = {
    "none": 60,
    "sign_only": 40,
    "meter_only": 25,
    "curb_only": 25,
    "sign_meter": 20,
    "sign_curb": 20,
    "meter_curb": 15,
    "sign_meter_curb": 15,
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_existing_image(image_key, candidate_dirs):
    for d in candidate_dirs:
        p = d / f"{image_key}.jpg"
        if p.exists():
            return str(p)
    return ""


def contains_label_recursive(obj, target_label):
    if isinstance(obj, dict):
        if obj.get("label") == target_label:
            return True
        return any(contains_label_recursive(v, target_label) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_label_recursive(x, target_label) for x in obj)
    return False


def build_sign_pools(mtsd_root, image_binary_csv):
    rows = read_csv(image_binary_csv)

    image_dirs = [
        mtsd_root / "data/images/train",
        mtsd_root / "data/images/val",
        mtsd_root / "dataset/mtsd/images",
    ]

    sign_pos = []
    sign_neg = []

    for r in rows:
        key = r["image_key"]
        img_path = find_existing_image(key, image_dirs)
        if not img_path:
            continue

        item = {
            "image_id": f"mtsd_{key}",
            "source_dataset": "MTSD",
            "image_key": key,
            "image_path": img_path,
            "gt_sign": int(r["binary_label"]),
            "gt_meter": 0,
            "gt_curb_color": 0,
            "dominant_color": "",
            "cue_pool": "sign_pos" if int(r["binary_label"]) == 1 else "none_candidate",
        }

        if int(r["binary_label"]) == 1:
            sign_pos.append(item)
        else:
            sign_neg.append(item)

    return sign_pos, sign_neg


def build_meter_pool(vistas_root, split="validation"):
    # Prefer the 2,000-image Kaggle eval subset if present, because it has matched images+polygons.
    candidate_roots = [
        vistas_root / "kaggle_meter_eval_upload" / split,
        vistas_root / split,
    ]

    meter_pos = []
    meter_neutral = []

    for root in candidate_roots:
        images_dir = root / "images"
        polygons_dir = root / "v2.0/polygons"
        if images_dir.exists() and polygons_dir.exists():
            break
    else:
        raise FileNotFoundError("Could not find Vistas images + v2.0/polygons for meter pool.")

    for ann_path in sorted(polygons_dir.glob("*.json")):
        image_id = ann_path.stem
        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        try:
            ann = json.loads(ann_path.read_text())
        except Exception:
            continue

        has_meter = contains_label_recursive(ann, "object--parking-meter")

        item = {
            "image_id": f"vistas_{image_id}",
            "source_dataset": "Vistas",
            "image_key": image_id,
            "image_path": str(img_path),
            "gt_sign": 0,
            "gt_meter": int(has_meter),
            "gt_curb_color": 0,
            "dominant_color": "",
            "cue_pool": "meter_pos" if has_meter else "none_candidate",
        }

        if has_meter:
            meter_pos.append(item)
        else:
            meter_neutral.append(item)

    return meter_pos, meter_neutral


def build_curb_pools(curb_csv):
    rows = read_csv(curb_csv)

    curb_color_pos = []
    curb_neutral = []

    for r in rows:
        image_id = r["image_id"]
        img_path = r["image_path"]
        dominant_color = r["dominant_color"].strip().lower()
        curb_present = int(float(r["curb_present"]))

        strong_curb = curb_present == 1 and dominant_color in STRONG_CURB_COLORS

        item = {
            "image_id": f"curb_{image_id}",
            "source_dataset": "VistasCurb",
            "image_key": image_id,
            "image_path": img_path,
            "gt_sign": 0,
            "gt_meter": 0,
            "gt_curb_color": int(strong_curb),
            "dominant_color": dominant_color,
            "cue_pool": "curb_color_pos" if strong_curb else "none_candidate",
        }

        if strong_curb:
            curb_color_pos.append(item)
        else:
            curb_neutral.append(item)

    return curb_color_pos, curb_neutral


def sample_from_pool(pool, k, rng, pool_name):
    if k == 0:
        return []
    if len(pool) == 0:
        raise ValueError(f"Pool '{pool_name}' is empty; cannot sample {k} images.")
    if len(pool) >= k:
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def segment_requirements(segment_type):
    req = {"sign": 0, "meter": 0, "curb": 0}

    if "sign" in segment_type:
        req["sign"] = 1
    if "meter" in segment_type:
        req["meter"] = 1
    if "curb" in segment_type:
        req["curb"] = 1

    return req


def make_segments(pools, type_counts, segment_size, seed):
    rng = random.Random(seed)

    image_pool_rows = []
    seen_pool_image_ids = set()

    for pool_name, pool_items in pools.items():
        for item in pool_items:
            if item["image_id"] not in seen_pool_image_ids:
                image_pool_rows.append(item)
                seen_pool_image_ids.add(item["image_id"])

    segment_rows = []
    segment_image_rows = []

    seg_counter = 0

    for segment_type, n_segments in type_counts.items():
        req = segment_requirements(segment_type)

        for _ in range(n_segments):
            seg_id = f"seg_{seg_counter:05d}"
            seg_counter += 1

            chosen = []
            chosen += sample_from_pool(pools["sign_pos"], req["sign"], rng, "sign_pos")
            chosen += sample_from_pool(pools["meter_pos"], req["meter"], rng, "meter_pos")
            chosen += sample_from_pool(pools["curb_color_pos"], req["curb"], rng, "curb_color_pos")

            remaining = segment_size - len(chosen)
            if remaining < 0:
                raise ValueError(f"segment_size={segment_size} too small for type {segment_type}")

            chosen_ids = {x["image_id"] for x in chosen}
            none_candidates = [x for x in pools["none"] if x["image_id"] not in chosen_ids]
            chosen += sample_from_pool(none_candidates, remaining, rng, "none")

            rng.shuffle(chosen)

            gt_sign_count = sum(int(x["gt_sign"]) for x in chosen)
            gt_meter_count = sum(int(x["gt_meter"]) for x in chosen)
            gt_curb_count = sum(int(x["gt_curb_color"]) for x in chosen)

            segment_label = 0 if segment_type == "none" else 1

            segment_rows.append({
                "segment_id": seg_id,
                "segment_type": segment_type,
                "segment_label": segment_label,
                "segment_size": len(chosen),
                "gt_sign_count": gt_sign_count,
                "gt_meter_count": gt_meter_count,
                "gt_curb_color_count": gt_curb_count,
                "has_sign": int(gt_sign_count > 0),
                "has_meter": int(gt_meter_count > 0),
                "has_curb_color": int(gt_curb_count > 0),
            })

            for order, item in enumerate(chosen):
                segment_image_rows.append({
                    "segment_id": seg_id,
                    "order_in_segment": order,
                    "segment_type": segment_type,
                    "segment_label": segment_label,
                    **item,
                })

    return image_pool_rows, segment_rows, segment_image_rows


def parse_type_counts(type_counts_str):
    if not type_counts_str:
        return DEFAULT_TYPE_COUNTS

    counts = {}
    for part in type_counts_str.split(","):
        name, value = part.split(":")
        counts[name.strip()] = int(value.strip())
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-root", default=str(Path.home() / "Downloads/parking_aggregation_project"))
    parser.add_argument("--mtsd-root", default=str(Path.home() / "Downloads/mtsd_project"))
    parser.add_argument("--vistas-root", default=str(Path.home() / "Downloads/Mapillary_Vistas_Dataset"))
    parser.add_argument("--segment-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--type-counts",
        default="",
        help="Comma-separated counts, e.g. none:60,sign_only:40,meter_only:25,curb_only:25,sign_meter:20",
    )
    args = parser.parse_args()

    agg_root = Path(args.agg_root)
    mtsd_root = Path(args.mtsd_root)
    vistas_root = Path(args.vistas_root)

    image_binary_csv = agg_root / "metadata/image_binary_labels.csv"
    curb_csv = agg_root / "metadata/validation_curb_color_results.csv"

    out_dir = agg_root / "outputs/synthetic_segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building cue pools...")

    sign_pos, sign_neg = build_sign_pools(mtsd_root, image_binary_csv)
    meter_pos, meter_neutral = build_meter_pool(vistas_root)
    curb_pos, curb_neutral = build_curb_pools(curb_csv)

    # None pool intentionally mixes clean negatives / neutral examples.
    none_pool = sign_neg + meter_neutral + curb_neutral

    pools = {
        "sign_pos": sign_pos,
        "meter_pos": meter_pos,
        "curb_color_pos": curb_pos,
        "none": none_pool,
    }

    print("Pool sizes:")
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)}")

    type_counts = parse_type_counts(args.type_counts)

    print("\nSegment type counts:")
    for k, v in type_counts.items():
        print(f"  {k}: {v}")

    image_pool_rows, segment_rows, segment_image_rows = make_segments(
        pools=pools,
        type_counts=type_counts,
        segment_size=args.segment_size,
        seed=args.seed,
    )

    write_csv(
        out_dir / "image_pool.csv",
        image_pool_rows,
        fieldnames=[
            "image_id", "source_dataset", "image_key", "image_path",
            "gt_sign", "gt_meter", "gt_curb_color", "dominant_color", "cue_pool"
        ],
    )

    write_csv(
        out_dir / "synthetic_segments.csv",
        segment_rows,
        fieldnames=[
            "segment_id", "segment_type", "segment_label", "segment_size",
            "gt_sign_count", "gt_meter_count", "gt_curb_color_count",
            "has_sign", "has_meter", "has_curb_color",
        ],
    )

    write_csv(
        out_dir / "synthetic_segment_images.csv",
        segment_image_rows,
        fieldnames=[
            "segment_id", "order_in_segment", "segment_type", "segment_label",
            "image_id", "source_dataset", "image_key", "image_path",
            "gt_sign", "gt_meter", "gt_curb_color", "dominant_color", "cue_pool",
        ],
    )

    summary = {
        "segment_size": args.segment_size,
        "seed": args.seed,
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "type_counts": type_counts,
        "n_segments": len(segment_rows),
        "n_segment_images": len(segment_image_rows),
        "segment_type_distribution": dict(Counter(r["segment_type"] for r in segment_rows)),
        "segment_label_distribution": dict(Counter(str(r["segment_label"]) for r in segment_rows)),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f"  {out_dir / 'image_pool.csv'}")
    print(f"  {out_dir / 'synthetic_segments.csv'}")
    print(f"  {out_dir / 'synthetic_segment_images.csv'}")
    print(f"  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
