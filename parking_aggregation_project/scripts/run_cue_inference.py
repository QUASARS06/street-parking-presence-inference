#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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


def classify_curb_color(image_rgb, mask_binary):
    result = {
        "curb_present": 0,
        "curb_pixel_count": 0,
        "dominant_color": "unknown",
        "dominant_confidence": 0.0,
        "red_score": 0.0,
        "yellow_score": 0.0,
        "green_score": 0.0,
        "white_score": 0.0,
        "gray_score": 0.0,
        "unknown_score": 1.0,
    }

    edges = cv2.Canny(mask_binary, 50, 150)
    ys, xs = np.where(edges > 0)

    if len(xs) < 30:
        ys, xs = np.where(mask_binary > 0)

    pixel_count = len(xs)
    result["curb_pixel_count"] = int(pixel_count)

    if pixel_count < 50:
        return result

    result["curb_present"] = 1

    pixels = image_rgb[ys, xs]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    keep = v >= 40
    h, s, v = h[keep], s[keep], v[keep]

    if len(h) == 0:
        return result

    red = ((h <= 10) | (h >= 170)) & (s >= 60)
    yellow = (h >= 15) & (h <= 40) & (s >= 60)
    green = (h >= 40) & (h <= 95) & (s >= 60)
    white = (s <= 40) & (v >= 200)
    gray = (s <= 50) & (v >= 60) & (v < 200)

    assigned = red | yellow | green | white | gray
    unknown = ~assigned
    total = len(h)

    scores = {
        "red": float(red.sum() / total),
        "yellow": float(yellow.sum() / total),
        "green": float(green.sum() / total),
        "white": float(white.sum() / total),
        "gray": float(gray.sum() / total),
        "unknown": float(unknown.sum() / total),
    }

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top1_color, top1_score = sorted_scores[0]
    _, top2_score = sorted_scores[1]

    conf_threshold = 0.35
    margin_threshold = 0.15

    if top1_color != "unknown" and top1_score >= conf_threshold and (top1_score - top2_score) >= margin_threshold:
        dominant_color = top1_color
        dominant_confidence = top1_score
    else:
        dominant_color = "unknown"
        dominant_confidence = scores["unknown"]

    result.update({
        "dominant_color": dominant_color,
        "dominant_confidence": float(dominant_confidence),
        "red_score": scores["red"],
        "yellow_score": scores["yellow"],
        "green_score": scores["green"],
        "white_score": scores["white"],
        "gray_score": scores["gray"],
        "unknown_score": scores["unknown"],
    })

    return result


def run_curb_inference(image_path, model, device, img_size=512, threshold=0.5):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    mask = clean_mask(mask)

    return classify_curb_color(img_rgb, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-root", default=str(Path.home() / "Downloads/parking_aggregation_project"))
    parser.add_argument("--sign-imgsz", type=int, default=640)
    parser.add_argument("--sign-conf", type=float, default=0.05)
    parser.add_argument("--meter-imgsz", type=int, default=1280)
    parser.add_argument("--meter-conf", type=float, default=0.05)
    parser.add_argument("--curb-img-size", type=int, default=512)
    parser.add_argument("--curb-threshold", type=float, default=0.5)
    args = parser.parse_args()

    agg_root = Path(args.agg_root)

    input_csv = agg_root / "outputs/synthetic_segments/synthetic_segment_images.csv"
    output_csv = agg_root / "outputs/synthetic_segments/synthetic_segment_image_predictions.csv"

    sign_model_path = agg_root / "checkpoints/parking_sign_best.pt"
    meter_model_path = agg_root / "checkpoints/yolo11x.pt"
    curb_model_path = agg_root / "checkpoints/curb_best.pt"

    rows = read_csv(input_csv)

    device = get_device()
    print("Device:", device)

    print("Loading models...")
    sign_model = YOLO(str(sign_model_path))
    meter_model = YOLO(str(meter_model_path))

    torch_device = torch.device(device)
    curb_model = build_curb_model().to(torch_device)
    curb_model.load_state_dict(torch.load(curb_model_path, map_location=torch_device))
    curb_model.eval()

    out_rows = []

    print(f"Running inference on {len(rows)} images...")

    for idx, r in enumerate(rows, start=1):
        image_path = Path(r["image_path"])

        sign_score = 0.0
        sign_n = 0
        meter_score = 0.0
        meter_n = 0

        try:
            sign_res = sign_model.predict(
                source=str(image_path),
                imgsz=args.sign_imgsz,
                conf=args.sign_conf,
                iou=0.5,
                save=False,
                verbose=False,
            )[0]
            if sign_res.boxes is not None and len(sign_res.boxes) > 0:
                confs = sign_res.boxes.conf.cpu().numpy()
                sign_score = float(confs.max())
                sign_n = int(len(confs))
        except Exception as e:
            print(f"[WARN] sign failed for {image_path}: {e}")

        try:
            meter_res = meter_model.predict(
                source=str(image_path),
                classes=[12],
                imgsz=args.meter_imgsz,
                conf=args.meter_conf,
                iou=0.5,
                save=False,
                verbose=False,
            )[0]
            if meter_res.boxes is not None and len(meter_res.boxes) > 0:
                confs = meter_res.boxes.conf.cpu().numpy()
                meter_score = float(confs.max())
                meter_n = int(len(confs))
        except Exception as e:
            print(f"[WARN] meter failed for {image_path}: {e}")

        curb_info = run_curb_inference(
            image_path=image_path,
            model=curb_model,
            device=torch_device,
            img_size=args.curb_img_size,
            threshold=args.curb_threshold,
        )

        if curb_info is None:
            curb_info = {
                "curb_present": 0,
                "curb_pixel_count": 0,
                "dominant_color": "unknown",
                "dominant_confidence": 0.0,
                "red_score": 0.0,
                "yellow_score": 0.0,
                "green_score": 0.0,
                "white_score": 0.0,
                "gray_score": 0.0,
                "unknown_score": 1.0,
            }

        strong_color_score = max(
            float(curb_info["yellow_score"]),
            float(curb_info["green_score"]),
            float(curb_info["white_score"]),
        )

        out = dict(r)
        out.update({
            "pred_sign_score": sign_score,
            "pred_sign_n": sign_n,
            "pred_meter_score": meter_score,
            "pred_meter_n": meter_n,
            "pred_curb_present": curb_info["curb_present"],
            "pred_curb_pixel_count": curb_info["curb_pixel_count"],
            "pred_curb_color": curb_info["dominant_color"],
            "pred_curb_color_confidence": curb_info["dominant_confidence"],
            "pred_curb_red_score": curb_info["red_score"],
            "pred_curb_yellow_score": curb_info["yellow_score"],
            "pred_curb_green_score": curb_info["green_score"],
            "pred_curb_white_score": curb_info["white_score"],
            "pred_curb_gray_score": curb_info["gray_score"],
            "pred_curb_unknown_score": curb_info["unknown_score"],
            "pred_curb_strong_color_score": strong_color_score,
        })

        out_rows.append(out)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(rows)}")

    fieldnames = list(out_rows[0].keys())
    write_csv(output_csv, out_rows, fieldnames)

    print("Wrote:", output_csv)


if __name__ == "__main__":
    main()
