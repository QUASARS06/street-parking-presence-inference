#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


STRONG_CURB_COLORS = {"red", "yellow", "green", "white"}


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

    if top1_color != "unknown" and top1_score >= 0.35 and (top1_score - top2_score) >= 0.15:
        dominant_color = top1_color
        dominant_conf = top1_score
    else:
        dominant_color = "unknown"
        dominant_conf = scores["unknown"]

    result.update({
        "dominant_color": dominant_color,
        "dominant_confidence": float(dominant_conf),
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


def prepare_manual_image_rows(manual_root):
    segments_csv = manual_root / "segments.csv"
    rows = read_csv(segments_csv)

    image_rows = []

    for seg in rows:
        seg_id = seg["segment_id"]
        seg_label = int(seg["segment_label"])
        gmap_link = seg.get("gmap_link", "")
        notes = seg.get("notes", "")

        seg_dir = manual_root / "segments" / seg_id
        images = sorted(list(seg_dir.glob("*.jpg")) + list(seg_dir.glob("*.png")) + list(seg_dir.glob("*.jpeg")))

        if len(images) == 0:
            print(f"[WARN] No images found for {seg_id}: {seg_dir}")
            continue

        for i, img_path in enumerate(images):
            image_rows.append({
                "segment_id": seg_id,
                "segment_label": seg_label,
                "segment_type": "manual",
                "order_in_segment": i,
                "image_id": f"{seg_id}_img_{i}",
                "image_path": str(img_path),
                "gmap_link": gmap_link,
                "notes": notes,
            })

    return image_rows


def aggregate_segments(pred_rows):
    by_seg = {}

    for r in pred_rows:
        by_seg.setdefault(r["segment_id"], []).append(r)

    segment_rows = []

    for seg_id, rows in by_seg.items():
        sign_scores = np.array([float(r["pred_sign_score"]) for r in rows])
        meter_scores = np.array([float(r["pred_meter_score"]) for r in rows])
        curb_scores = np.array([float(r["pred_curb_strong_color_score"]) for r in rows])

        sign_max = float(sign_scores.max())
        meter_max = float(meter_scores.max())
        curb_max = float(curb_scores.max())

        combined_score = max(
            sign_max,
            0.6 * meter_max,
            0.4 * curb_max,
        )

        single_first_score = float(sign_scores[0])

        segment_rows.append({
            "segment_id": seg_id,
            "segment_label": rows[0]["segment_label"],
            "n_images": len(rows),
            "gmap_link": rows[0].get("gmap_link", ""),
            "notes": rows[0].get("notes", ""),
            "single_first_sign_score": single_first_score,
            "sign_max": sign_max,
            "meter_max": meter_max,
            "curb_max": curb_max,
            "combined_score": combined_score,
            "pred_label_at_015": int(combined_score >= 0.15),
            "single_pred_at_015": int(single_first_score >= 0.15),
            "best_sign_image": max(rows, key=lambda x: float(x["pred_sign_score"]))["image_id"],
            "best_meter_image": max(rows, key=lambda x: float(x["pred_meter_score"]))["image_id"],
            "best_curb_image": max(rows, key=lambda x: float(x["pred_curb_strong_color_score"]))["image_id"],
        })

    return sorted(segment_rows, key=lambda x: x["segment_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-root", default=str(Path.home() / "Downloads/parking_aggregation_project"))
    parser.add_argument("--manual-root", default=str(Path.home() / "Downloads/manual_segments_dataset"))
    parser.add_argument("--sign-imgsz", type=int, default=640)
    parser.add_argument("--sign-conf", type=float, default=0.05)
    parser.add_argument("--meter-imgsz", type=int, default=1280)
    parser.add_argument("--meter-conf", type=float, default=0.05)
    parser.add_argument("--curb-img-size", type=int, default=512)
    parser.add_argument("--curb-threshold", type=float, default=0.5)
    args = parser.parse_args()

    agg_root = Path(args.agg_root)
    manual_root = Path(args.manual_root)

    out_dir = manual_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sign_model_path = agg_root / "checkpoints/parking_sign_best.pt"
    meter_model_path = agg_root / "checkpoints/yolo11x.pt"
    curb_model_path = agg_root / "checkpoints/curb_best.pt"

    device = get_device()
    print("Device:", device)

    print("Preparing manual image rows...")
    image_rows = prepare_manual_image_rows(manual_root)

    if len(image_rows) == 0:
        raise RuntimeError("No manual images found. Check folder structure.")

    write_csv(
        out_dir / "manual_segment_images.csv",
        image_rows,
        ["segment_id", "segment_label", "segment_type", "order_in_segment", "image_id", "image_path", "gmap_link", "notes"],
    )

    print("Loading models...")
    sign_model = YOLO(str(sign_model_path))
    meter_model = YOLO(str(meter_model_path))

    torch_device = torch.device(device)
    curb_model = build_curb_model().to(torch_device)
    curb_model.load_state_dict(torch.load(curb_model_path, map_location=torch_device))
    curb_model.eval()

    pred_rows = []

    for r in tqdm(image_rows, desc="Running manual cue inference"):
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

        strong_curb_score = max(
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
            "pred_curb_strong_color_score": strong_curb_score,
        })
        pred_rows.append(out)

    write_csv(
        out_dir / "manual_predictions.csv",
        pred_rows,
        list(pred_rows[0].keys()),
    )

    segment_rows = aggregate_segments(pred_rows)

    write_csv(
        out_dir / "manual_segment_scores.csv",
        segment_rows,
        [
            "segment_id", "segment_label", "n_images", "gmap_link", "notes",
            "single_first_sign_score", "sign_max", "meter_max", "curb_max",
            "combined_score", "pred_label_at_015", "single_pred_at_015",
            "best_sign_image", "best_meter_image", "best_curb_image",
        ],
    )

    print("\nWrote:")
    print(out_dir / "manual_segment_images.csv")
    print(out_dir / "manual_predictions.csv")
    print(out_dir / "manual_segment_scores.csv")

    print("\nSegment scores:")
    for r in segment_rows:
        print(
            f"{r['segment_id']} | label={r['segment_label']} | "
            f"single={float(r['single_first_sign_score']):.3f} | "
            f"sign={float(r['sign_max']):.3f} | "
            f"meter={float(r['meter_max']):.3f} | "
            f"curb={float(r['curb_max']):.3f} | "
            f"combined={float(r['combined_score']):.3f} | "
            f"pred={r['pred_label_at_015']}"
        )


if __name__ == "__main__":
    main()
