#!/usr/bin/env python3
import os
import json
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CURB_LABEL = "construction--barrier--curb"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare binary curb segmentation dataset from Mapillary Vistas polygons."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to Mapillary Vistas root containing training/ and validation/ folders",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output folder for prepared curb segmentation dataset",
    )
    parser.add_argument(
        "--image-mode",
        type=str,
        default="symlink",
        choices=["copy", "symlink"],
        help="Whether to copy images or create symlinks",
    )
    parser.add_argument(
        "--mask-value",
        type=int,
        default=255,
        choices=[1, 255],
        help="Foreground value to write in masks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def polygon_to_int32(poly):
    arr = np.array(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        return None
    return np.round(arr).astype(np.int32)


def create_binary_curb_mask(json_data, height, width, fg_value=255):
    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in json_data.get("objects", []):
        if obj.get("label") != CURB_LABEL:
            continue

        poly = polygon_to_int32(obj.get("polygon", []))
        if poly is None:
            continue

        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [poly], color=fg_value)

    return mask


def copy_or_symlink_image(src: Path, dst: Path, mode: str, overwrite: bool = False):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image for copy: {src}")
        ok = cv2.imwrite(str(dst), img)
        if not ok:
            raise RuntimeError(f"Failed to write copied image: {dst}")
    else:
        os.symlink(src.resolve(), dst)


def process_split(split_name: str, data_root: Path, out_root: Path, image_mode: str, mask_value: int, overwrite: bool):
    img_dir = data_root / split_name / "images"
    poly_dir = data_root / split_name / "v2.0" / "polygons"

    out_img_dir = out_root / split_name / "images"
    out_mask_dir = out_root / split_name / "masks"
    out_meta_dir = out_root / "metadata"

    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)
    ensure_dir(out_meta_dir)

    json_files = sorted(poly_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No polygon JSON files found in {poly_dir}")

    csv_path = out_meta_dir / f"{split_name}_metadata.csv"
    txt_path = out_meta_dir / f"{split_name}_list.txt"

    rows = []

    for json_path in tqdm(json_files, desc=f"Preparing {split_name}"):
        stem = json_path.stem
        img_path = img_dir / f"{stem}.jpg"

        if not img_path.exists():
            print(f"[WARN] Missing image for {stem}, skipping")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image {img_path}, skipping")
            continue

        height, width = img.shape[:2]

        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[WARN] Failed to read {json_path}: {e}")
            continue

        mask = create_binary_curb_mask(data, height, width, fg_value=mask_value)

        curb_pixels = int((mask > 0).sum())
        total_pixels = int(mask.size)
        curb_fraction = curb_pixels / total_pixels
        has_curb = int(curb_pixels > 0)

        out_img_path = out_img_dir / f"{stem}.jpg"
        out_mask_path = out_mask_dir / f"{stem}.png"

        try:
            copy_or_symlink_image(img_path, out_img_path, image_mode, overwrite=overwrite)
        except Exception as e:
            print(f"[WARN] Failed image export for {stem}: {e}")
            continue

        if out_mask_path.exists() and not overwrite:
            pass
        else:
            ok = cv2.imwrite(str(out_mask_path), mask)
            if not ok:
                print(f"[WARN] Failed to write mask for {stem}")
                continue

        rows.append({
            "image_id": stem,
            "split": split_name,
            "image_path": str(out_img_path),
            "mask_path": str(out_mask_path),
            "height": height,
            "width": width,
            "has_curb": has_curb,
            "curb_pixels": curb_pixels,
            "total_pixels": total_pixels,
            "curb_fraction": f"{curb_fraction:.8f}",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "split",
                "image_path",
                "mask_path",
                "height",
                "width",
                "has_curb",
                "curb_pixels",
                "total_pixels",
                "curb_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_id']}\n")

    n_total = len(rows)
    n_pos = sum(int(r["has_curb"]) for r in rows)
    print(f"\n[{split_name}] done")
    print(f"  images written: {n_total}")
    print(f"  positives:      {n_pos}")
    print(f"  negatives:      {n_total - n_pos}")
    print(f"  metadata:       {csv_path}")
    print(f"  id list:        {txt_path}")


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    ensure_dir(out_root)

    for split_name in ["training", "validation"]:
        process_split(
            split_name=split_name,
            data_root=data_root,
            out_root=out_root,
            image_mode=args.image_mode,
            mask_value=args.mask_value,
            overwrite=args.overwrite,
        )

    print("\nAll done.")
    print(f"Prepared dataset at: {out_root}")


if __name__ == "__main__":
    main()