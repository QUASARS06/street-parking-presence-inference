#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def draw_text(img, text, y):
    cv2.putText(img, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)


def annotate_image(row):
    img = cv2.imread(row["image_path"])
    if img is None:
        return None

    # Add text annotations
    draw_text(img, f"Sign: {row['pred_sign_score']:.2f}", 25)
    draw_text(img, f"Meter: {row['pred_meter_score']:.2f}", 50)
    draw_text(img, f"Curb: {row['pred_curb_strong_color_score']:.2f}", 75)

    return img


def make_collage(images, cols=5):
    h, w = images[0].shape[:2]

    resized = [cv2.resize(im, (w, h)) for im in images]

    rows = []
    for i in range(0, len(resized), cols):
        row = np.hstack(resized[i:i+cols])
        rows.append(row)

    collage = np.vstack(rows)
    return collage


def main():
    root = Path.home() / "Downloads/manual_segments_dataset"
    pred_csv = root / "outputs/manual_predictions.csv"

    df = pd.read_csv(pred_csv)

    out_root = root / "outputs/visualizations"
    out_root.mkdir(parents=True, exist_ok=True)

    for seg_id, group in tqdm(df.groupby("segment_id"), desc="Visualizing"):
        seg_dir = out_root / seg_id
        seg_dir.mkdir(parents=True, exist_ok=True)

        annotated_imgs = []

        for _, row in group.iterrows():
            img = annotate_image(row)
            if img is None:
                continue

            out_path = seg_dir / f"{row['image_id']}_annotated.jpg"
            cv2.imwrite(str(out_path), img)
            annotated_imgs.append(img)

        if len(annotated_imgs) > 0:
            collage = make_collage(annotated_imgs)
            cv2.imwrite(str(seg_dir / "collage.jpg"), collage)

    print("Saved visualizations to:", out_root)


if __name__ == "__main__":
    main()
