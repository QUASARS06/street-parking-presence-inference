#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# -----------------------------
# Device
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Model
# -----------------------------
def build_model(encoder="resnet34"):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1
    )


# -----------------------------
# Transform
# -----------------------------
def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# -----------------------------
# Mask cleaning
# -----------------------------
def clean_mask(mask, min_area=25):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


# -----------------------------
# Color classification
# -----------------------------
def classify_curb_color(image_rgb, mask_binary):

    result = {
        "curb_present": False,
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

    # ---- EDGE-BASED PIXELS ----
    edges = cv2.Canny(mask_binary, 50, 150)
    ys, xs = np.where(edges > 0)

    pixel_count = len(xs)

    # fallback to full mask if too few
    if pixel_count < 30:
        ys, xs = np.where(mask_binary > 0)
        pixel_count = len(xs)

    result["curb_pixel_count"] = int(pixel_count)

    if pixel_count < 50:
        return result

    result["curb_present"] = True

    pixels = image_rgb[ys, xs]
    hsv = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_RGB2HSV).reshape(-1,3)

    h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]

    # remove dark pixels
    keep = v >= 40
    h, s, v = h[keep], s[keep], v[keep]

    if len(h) == 0:
        return result

    # ---- COLOR RULES ----
    red = ((h <= 10) | (h >= 170)) & (s >= 60)
    yellow = (h >= 15) & (h <= 40) & (s >= 60)
    green = (h >= 40) & (h <= 95) & (s >= 60)
    white = (s <= 40) & (v >= 200)
    gray = (s <= 50) & (v >= 60) & (v < 200)

    assigned = red | yellow | green | white | gray
    unknown = ~assigned

    total = len(h)

    scores = {
        "red": red.sum()/total,
        "yellow": yellow.sum()/total,
        "green": green.sum()/total,
        "white": white.sum()/total,
        "gray": gray.sum()/total,
        "unknown": unknown.sum()/total,
    }

    # ---- TOP-2 MARGIN RULE ----
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top1_color, top1_score = sorted_scores[0]
    top2_color, top2_score = sorted_scores[1]

    CONF_THRESHOLD = 0.35
    MARGIN_THRESHOLD = 0.15

    if top1_color != "unknown":
        if top1_score < CONF_THRESHOLD or (top1_score - top2_score) < MARGIN_THRESHOLD:
            dominant_color = "unknown"
            dominant_conf = scores["unknown"]
        else:
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

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/infer")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load image
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Failed to load image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # transform
    transform = get_transform(args.img_size)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    # model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # inference
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()

    mask_small = (prob > args.threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = clean_mask(mask)

    # color
    color_info = classify_curb_color(img_rgb, mask)

    # overlay
    overlay = img_rgb.copy()
    overlay[mask > 0] = [0,255,0]

    # save outputs
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    cv2.imwrite(str(out_dir / "overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("\n=== RESULT ===")
    for k,v in color_info.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(out_dir / "mask.png")
    print(out_dir / "overlay.jpg")


if __name__ == "__main__":
    main()