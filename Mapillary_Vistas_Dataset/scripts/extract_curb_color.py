#!/usr/bin/env python3
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Dataset
# -----------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image_path = row["image_path"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        if self.transform is not None:
            aug = self.transform(image=image_rgb)
            image_tensor = aug["image"]
        else:
            image_tensor = image_rgb

        return {
            "image_id": image_id,
            "image_path": image_path,
            "orig_h": h,
            "orig_w": w,
            "image_rgb": image_rgb,
            "image_tensor": image_tensor,
        }


def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# -----------------------------
# Model
# -----------------------------
def build_model(encoder="resnet34"):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    return model


# -----------------------------
# Mask post-processing
# -----------------------------
def clean_mask(mask, min_component_area=20, kernel_size=3):
    """
    mask: uint8 binary mask in {0,255}
    """
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == lab] = 255

    return cleaned


# -----------------------------
# Color extraction
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
# Visualization
# -----------------------------
def make_overlay(image_rgb, mask_binary, color_label):
    overlay = image_rgb.copy()

    # curb mask in green
    overlay[mask_binary > 0] = [0, 255, 0]

    panel = overlay.copy()
    cv2.putText(
        panel,
        f"color={color_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return panel


# -----------------------------
# Main processing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Predict curb masks and extract curb color.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to curb_segmentation root (with metadata CSVs)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained curb segmentation best.pt")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["training", "validation"])
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=25)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-viz", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    masks_dir = out_dir / args.split / "pred_masks"
    viz_dir = out_dir / args.split / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_root / "metadata" / f"{args.split}_metadata.csv"
    dataset = ImageOnlyDataset(
        csv_path=csv_path,
        transform=get_transform(args.img_size)
    )

    # Custom collate because we keep image_rgb in numpy
    def collate_fn(batch):
        image_ids = [x["image_id"] for x in batch]
        image_paths = [x["image_path"] for x in batch]
        orig_h = [x["orig_h"] for x in batch]
        orig_w = [x["orig_w"] for x in batch]
        image_rgbs = [x["image_rgb"] for x in batch]
        image_tensors = torch.stack([x["image_tensor"] for x in batch], dim=0)
        return {
            "image_id": image_ids,
            "image_path": image_paths,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "image_rgb": image_rgbs,
            "image_tensor": image_tensors,
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(args.encoder).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    rows = []
    viz_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Processing {args.split}"):
            images = batch["image_tensor"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

            for i in range(len(batch["image_id"])):
                image_id = batch["image_id"][i]
                image_path = batch["image_path"][i]
                orig_h = batch["orig_h"][i]
                orig_w = batch["orig_w"][i]
                image_rgb = batch["image_rgb"][i]

                prob_map = probs[i]
                pred_small = (prob_map > args.threshold).astype(np.uint8) * 255

                pred_mask = cv2.resize(
                    pred_small,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )

                pred_mask = clean_mask(pred_mask, min_component_area=args.min_mask_pixels, kernel_size=3)

                color_info = classify_curb_color(image_rgb, pred_mask)

                mask_out_path = masks_dir / f"{image_id}.png"
                cv2.imwrite(str(mask_out_path), pred_mask)

                if args.save_viz and viz_count < args.max_viz:
                    overlay = make_overlay(image_rgb, pred_mask, color_info["dominant_color"])
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(viz_dir / f"{image_id}.jpg"), overlay_bgr)
                    viz_count += 1

                rows.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "mask_path": str(mask_out_path),
                    "curb_present": int(color_info["curb_present"]),
                    "curb_pixel_count": color_info["curb_pixel_count"],
                    "dominant_color": color_info["dominant_color"],
                    "dominant_confidence": color_info["dominant_confidence"],
                    "red_score": color_info["red_score"],
                    "yellow_score": color_info["yellow_score"],
                    "green_score": color_info["green_score"],
                    "white_score": color_info["white_score"],
                    "gray_score": color_info["gray_score"],
                    "unknown_score": color_info["unknown_score"],
                })

    df = pd.DataFrame(rows)
    csv_out = out_dir / f"{args.split}_curb_color_results.csv"
    df.to_csv(csv_out, index=False)

    print(f"\nSaved results to: {csv_out}")
    print(df["dominant_color"].value_counts(dropna=False))
    print("\nDone.")


if __name__ == "__main__":
    main()