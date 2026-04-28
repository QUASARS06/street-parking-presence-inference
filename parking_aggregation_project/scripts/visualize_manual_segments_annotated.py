#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_curb_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def clean_mask(mask, min_area=25):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def predict_curb_mask(image_bgr, model, device, img_size=512, threshold=0.5):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean_mask(mask)


def overlay_curb(img, mask):
    if mask is None or mask.max() == 0:
        return img

    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(overlay, 0.35, img, 0.65, 0)


def draw_yolo_boxes(img, result, label, color):
    if result.boxes is None or len(result.boxes) == 0:
        return img

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
        conf = float(box.conf.cpu().numpy()[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        text = f"{label} {conf:.2f}"
        y_text = max(25, y1 - 8)
        cv2.putText(
            img,
            text,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def add_header(img, row):
    header_h = 70
    h, w = img.shape[:2]

    canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
    canvas[header_h:] = img

    text1 = f"{row['segment_id']} | {row['image_id']}"
    text2 = (
        f"sign={float(row['pred_sign_score']):.2f} "
        f"meter={float(row['pred_meter_score']):.2f} "
        f"curb={float(row['pred_curb_strong_color_score']):.2f} "
        f"curb_color={row['pred_curb_color']}"
    )

    cv2.putText(canvas, text1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, text2, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return canvas


def make_collage(images, thumb_w=420):
    thumbs = []

    for img in images:
        h, w = img.shape[:2]
        scale = thumb_w / w
        thumb_h = int(h * scale)
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h)))

    max_h = max(t.shape[0] for t in thumbs)
    padded = []

    for t in thumbs:
        h, w = t.shape[:2]
        canvas = np.full((max_h, thumb_w, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = t
        padded.append(canvas)

    return np.hstack(padded)


def main():
    agg_root = Path.home() / "Downloads/parking_aggregation_project"
    manual_root = Path.home() / "Downloads/manual_segments_dataset"

    pred_csv = manual_root / "outputs/manual_predictions.csv"
    out_root = manual_root / "outputs/annotated_visualizations"
    out_root.mkdir(parents=True, exist_ok=True)

    sign_model = YOLO(str(agg_root / "checkpoints/parking_sign_best.pt"))
    meter_model = YOLO(str(agg_root / "checkpoints/yolo11x.pt"))

    device = get_device()
    torch_device = torch.device(device)

    curb_model = build_curb_model().to(torch_device)
    curb_model.load_state_dict(
        torch.load(agg_root / "checkpoints/curb_best.pt", map_location=torch_device)
    )
    curb_model.eval()

    df = pd.read_csv(pred_csv)

    for seg_id, group in tqdm(df.groupby("segment_id"), desc="Creating annotated collages"):
        seg_dir = out_root / seg_id
        seg_dir.mkdir(parents=True, exist_ok=True)

        annotated = []

        for _, row in group.sort_values("order_in_segment").iterrows():
            img_path = Path(row["image_path"])
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"[WARN] Could not read {img_path}")
                continue

            # Curb overlay
            curb_mask = predict_curb_mask(img, curb_model, torch_device)
            img = overlay_curb(img, curb_mask)

            # Parking sign boxes
            sign_result = sign_model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.05,
                iou=0.5,
                save=False,
                verbose=False,
            )[0]
            img = draw_yolo_boxes(img, sign_result, "parking_sign", (255, 0, 0))

            # Parking meter boxes
            meter_result = meter_model.predict(
                source=str(img_path),
                classes=[12],
                imgsz=1280,
                conf=0.05,
                iou=0.5,
                save=False,
                verbose=False,
            )[0]
            img = draw_yolo_boxes(img, meter_result, "parking_meter", (255, 255, 0))

            img = add_header(img, row)

            out_path = seg_dir / f"{row['image_id']}_annotated.jpg"
            cv2.imwrite(str(out_path), img)

            annotated.append(img)

        if annotated:
            collage = make_collage(annotated)
            cv2.imwrite(str(seg_dir / "collage.jpg"), collage)

    print("Saved annotated visualizations to:")
    print(out_root)


if __name__ == "__main__":
    main()
