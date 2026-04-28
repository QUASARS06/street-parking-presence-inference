#!/usr/bin/env python3
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Dataset
# -----------------------------
class CurbSegDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        mask_path = row["mask_path"]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# -----------------------------
# Transforms
# -----------------------------
def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.2
        ),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# -----------------------------
# Losses / Metrics
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


def compute_batch_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean().item()

    return iou, dice


# -----------------------------
# Visualization
# -----------------------------
def save_prediction_samples(model, loader, device, out_dir, max_samples=12, threshold=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            masks_np = masks.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                if saved >= max_samples:
                    return

                img = images_np[i]
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

                gt = (masks_np[i, 0] * 255).astype(np.uint8)
                pr = (preds_np[i, 0] * 255).astype(np.uint8)

                overlay_gt = img.copy()
                overlay_pr = img.copy()

                overlay_gt[gt > 0] = [255, 0, 0]
                overlay_pr[pr > 0] = [0, 255, 0]

                panel = np.concatenate([
                    img,
                    cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(pr, cv2.COLOR_GRAY2RGB),
                    overlay_gt,
                    overlay_pr
                ], axis=1)

                panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"sample_{saved:03d}.jpg"), panel_bgr)
                saved += 1


# -----------------------------
# Train / Eval Loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)

        loss_bce = bce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        iou, dice = compute_batch_metrics(logits, masks)

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{iou:.4f}",
            "dice": f"{dice:.4f}"
        })

    n = len(loader)
    return running_loss / n, running_iou / n, running_dice / n


def eval_one_epoch(model, loader, bce_loss, dice_loss, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_bce + loss_dice

            iou, dice = compute_batch_metrics(logits, masks)

            running_loss += loss.item()
            running_iou += iou
            running_dice += dice

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou": f"{iou:.4f}",
                "dice": f"{dice:.4f}"
            })

    n = len(loader)
    return running_loss / n, running_iou / n, running_dice / n


# -----------------------------
# Checkpoint / Viz policy
# -----------------------------
def should_save_epoch_checkpoint(epoch, total_epochs, every_n=3, final_k=3):
    if epoch > total_epochs - final_k:
        return True
    return epoch % every_n == 0


def should_save_viz(epoch, total_epochs, every_n=3, final_k=3):
    if epoch > total_epochs - final_k:
        return True
    return epoch % every_n == 0


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train binary curb segmentation model.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to curb_segmentation root")
    parser.add_argument("--out-dir", type=str, default="outputs/curb_seg_run",
                        help="Where to save checkpoints and visualizations")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)

    train_csv = data_root / "metadata" / "training_metadata.csv"
    val_csv = data_root / "metadata" / "validation_metadata.csv"

    train_ds = CurbSegDataset(
        csv_path=train_csv,
        transform=get_train_transform(args.img_size)
    )
    val_ds = CurbSegDataset(
        csv_path=val_csv,
        transform=get_val_transform(args.img_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, bce_loss, dice_loss, device
        )
        val_loss, val_iou, val_dice = eval_one_epoch(
            model, val_loader, bce_loss, dice_loss, device
        )

        print(
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        })

        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        # always save last
        torch.save(model.state_dict(), out_dir / "checkpoints" / "last.pt")

        # save sparse epoch checkpoints initially, every epoch in final 3
        if should_save_epoch_checkpoint(epoch, args.epochs, every_n=3, final_k=3):
            torch.save(model.state_dict(), out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            print(f"Saved epoch checkpoint: epoch_{epoch:03d}.pt")

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            print(f"Saved new best model with val_dice={best_val_dice:.4f}")

            save_prediction_samples(
                model,
                val_loader,
                device,
                out_dir / "viz" / f"best_epoch_{epoch:03d}",
                max_samples=12,
                threshold=args.threshold
            )

        # periodic visualizations
        if should_save_viz(epoch, args.epochs, every_n=3, final_k=3):
            save_prediction_samples(
                model,
                val_loader,
                device,
                out_dir / "viz" / f"epoch_{epoch:03d}",
                max_samples=12,
                threshold=args.threshold
            )
            print(f"Saved visualization samples for epoch {epoch}")

    print("\nTraining complete.")
    print(f"Best val_dice: {best_val_dice:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()