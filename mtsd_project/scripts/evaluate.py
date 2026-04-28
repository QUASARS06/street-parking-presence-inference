"""
evaluate.py
-----------
Evaluates a trained YOLOv8 parking sign detector on val split.
Produces:
  - Standard YOLO metrics (mAP50, mAP50-95, precision, recall)
  - Image-level binary classification metrics (accuracy, F1, AUROC)
  - Per-threshold analysis to pick best confidence threshold
  - Optional saved visualizations of predictions on sample images
  - Per-image results CSV for segment aggregation experiments

Usage:
    python scripts/evaluate.py --weights best.pt
    python scripts/evaluate.py --weights best.pt --split val --device mps
    python scripts/evaluate.py --weights best.pt --split val --device cpu
    python scripts/evaluate.py --weights best.pt --split val --save-images
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_ROOT / "parking_signs.yaml"
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = PROJECT_ROOT / "dataset" / "mtsd" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "mtsd" / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Label definitions (must match prepare_dataset.py) ─────────────────────────
POSITIVE_LABELS_NA = {
    "regulatory--no-parking--g1",
    "regulatory--no-stopping--g15",
    "regulatory--no-stopping--g8",
    "regulatory--no-parking-or-no-stopping--g1",
    "regulatory--no-parking-or-no-stopping--g3",
    "regulatory--end-of-no-parking--g1",
    "complementary--tow-away-zone--g1",
    "information--parking--g1",
    "information--parking--g3",
    "information--parking--g5",
    "information--parking--g45",
    "regulatory--parking-restrictions--g2",
}

POSITIVE_LABELS_NON_NA = {
    "regulatory--no-parking--g2",
    "regulatory--no-parking--g5",
    "regulatory--no-stopping--g2",
    "regulatory--no-stopping--g4",
    "regulatory--no-stopping--g5",
    "regulatory--no-parking-or-no-stopping--g2",
    "information--no-parking--g3",
    "information--parking--g2",
    "information--parking--g6",
}

POSITIVE_LABELS = POSITIVE_LABELS_NA | POSITIVE_LABELS_NON_NA


def is_usable(obj):
    props = obj.get("properties", {})
    return not any([
        props.get("ambiguous", False),
        props.get("occluded", False),
        props.get("out-of-frame", False),
        props.get("dummy", False),
    ])


def get_ground_truth_label(key: str) -> tuple[int, bool]:
    """
    Returns:
      binary_label: 1 if image contains usable parking-sign object, else 0
      na_confident: True if image contains at least one NA-variant positive sign
    """
    ann_path = ANNOTATIONS_DIR / f"{key}.json"
    if not ann_path.exists():
        return 0, False

    with open(ann_path) as f:
        ann = json.load(f)

    usable = [o for o in ann.get("objects", []) if is_usable(o)]
    labels = [o.get("label", "") for o in usable]
    has_pos = any(l in POSITIVE_LABELS for l in labels)
    has_na = any(l in POSITIVE_LABELS_NA for l in labels)
    return int(has_pos), has_na


def load_split_keys(split: str) -> list[str]:
    f = SPLITS_DIR / f"{split}.txt"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def compute_binary_metrics(gt_labels, pred_scores, threshold):
    pred_labels = [1 if s >= threshold else 0 for s in pred_scores]
    tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 0 and p == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(len(gt_labels), 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_auroc(gt_labels, pred_scores):
    total_pos = sum(gt_labels)
    total_neg = len(gt_labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return float("nan")

    thresholds = sorted(set(pred_scores), reverse=True)
    tprs, fprs = [0.0], [0.0]

    for t in thresholds:
        pred = [1 if s >= t else 0 for s in pred_scores]
        tp = sum(1 for g, p in zip(gt_labels, pred) if g == 1 and p == 1)
        fp = sum(1 for g, p in zip(gt_labels, pred) if g == 0 and p == 1)
        tprs.append(tp / total_pos)
        fprs.append(fp / total_neg)

    tprs.append(1.0)
    fprs.append(1.0)

    auroc = sum(
        (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )
    return auroc


def find_best_threshold(gt_labels, pred_scores):
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        m = compute_binary_metrics(gt_labels, pred_scores, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = round(float(t), 2)
    return best_t, best_f1


def round_metric_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def clear_device_cache():
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def run(args):
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Torch MPS available : {torch.backends.mps.is_available()}")
    print(f"Requested device    : {args.device if args.device else 'auto'}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if args.split == "test":
        print("WARNING: MTSD test images/labels are typically not available locally.")
        print("Use val for evaluation unless you explicitly prepared a usable test split.\n")

    run_name = weights_path.parent.parent.name if weights_path.parent.parent.exists() else weights_path.stem
    split = args.split

    print("=" * 65)
    print(f"PARKING SIGN DETECTOR — EVALUATION ({split.upper()})")
    print("=" * 65)
    print(f"  Weights  : {weights_path}")
    print(f"  Split    : {split}")
    print()

    model = YOLO(str(weights_path))

    # ── 1. Standard YOLO metrics (box-level mAP) ──────────────────────────────
    print("Running YOLO validation (box-level metrics)...")
    yolo_metrics = model.val(
        data=str(YAML_PATH),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.5,
        device=args.device if args.device else None,
        plots=True,
        save_json=False,
        verbose=False,
    )

    map50 = yolo_metrics.box.map50
    map5095 = yolo_metrics.box.map
    prec = yolo_metrics.box.mp
    rec = yolo_metrics.box.mr

    print()
    print("  BOX-LEVEL METRICS (detection quality)")
    print(f"    mAP@50       : {map50:.4f}")
    print(f"    mAP@50-95    : {map5095:.4f}")
    print(f"    Precision    : {prec:.4f}")
    print(f"    Recall       : {rec:.4f}")

    # ── 2. Image-level binary metrics ─────────────────────────────────────────
    print()
    print("Running image-level binary inference...")

    keys = load_split_keys(split)
    images_dir = DATA_DIR / "images" / split
    valid_keys = [k for k in keys if (images_dir / f"{k}.jpg").exists()]

    if not valid_keys:
        print(f"  WARNING: No images found in {images_dir} for split '{split}'")
        return

    print(f"  Evaluating {len(valid_keys)} images...")
    print(f"  Chunk size: {args.chunk_size}")

    gt_labels = []
    gt_na_flags = []
    pred_scores = []
    per_image_rows = []

    result_map = {}
    n_chunks = (len(valid_keys) + args.chunk_size - 1) // args.chunk_size

    for chunk_idx, start in enumerate(range(0, len(valid_keys), args.chunk_size), start=1):
        chunk_keys = valid_keys[start:start + args.chunk_size]
        chunk_paths = [str(images_dir / f"{k}.jpg") for k in chunk_keys]

        print(f"  Predicting chunk {chunk_idx}/{n_chunks} ({len(chunk_keys)} images)...")

        results = model.predict(
            source=chunk_paths,
            imgsz=args.imgsz,
            conf=args.pred_conf,
            iou=0.5,
            device=args.device if args.device else None,
            verbose=False,
            stream=True,
        )

        for key, result in zip(chunk_keys, results):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.detach().cpu().numpy().tolist()
                max_conf = float(max(confs))
                n_det = len(confs)
            else:
                max_conf = 0.0
                n_det = 0
            result_map[key] = (max_conf, n_det)

        clear_device_cache()

    for key in valid_keys:
        gt, na_flag = get_ground_truth_label(key)
        max_conf, n_det = result_map.get(key, (0.0, 0))

        gt_labels.append(gt)
        gt_na_flags.append(na_flag)
        pred_scores.append(max_conf)

        per_image_rows.append({
            "image_key": key,
            "gt_label": gt,
            "na_confident": int(na_flag),
            "max_confidence": round(max_conf, 4),
            "n_detections": n_det,
        })

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print()
    print("  IMAGE-LEVEL BINARY METRICS (for aggregation pipeline)")
    print()
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("  " + "-" * 80)

    sweep_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    for t in sweep_thresholds:
        m = compute_binary_metrics(gt_labels, pred_scores, t)
        print(
            f"  {t:>10.2f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['accuracy']:>10.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>6}"
        )

    best_t, best_f1 = find_best_threshold(gt_labels, pred_scores)
    best_m = compute_binary_metrics(gt_labels, pred_scores, best_t)
    auroc = compute_auroc(gt_labels, pred_scores)

    print()
    print(f"  Best threshold (max F1)  : {best_t:.2f}")
    print(f"  Best F1                  : {best_f1:.4f}")
    print(f"  AUROC                    : {auroc:.4f}" if not np.isnan(auroc) else "  AUROC                    : nan")

    # ── Ground truth distribution ─────────────────────────────────────────────
    total = len(gt_labels)
    n_pos = sum(gt_labels)
    n_neg = total - n_pos
    print()
    print(f"  Ground truth distribution  : {n_pos} positive, {n_neg} negative ({total} total)")

    # ── Save per-image CSV ────────────────────────────────────────────────────
    csv_path = OUTPUTS_DIR / f"eval_{split}_{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "run": run_name,
        "split": split,
        "weights": str(weights_path),
        "n_images": total,
        "box_metrics": {
            "mAP50": round(float(map50), 4),
            "mAP50_95": round(float(map5095), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        },
        "image_level_metrics": {
            "best_threshold": best_t,
            "best_f1": round(best_f1, 4),
            "auroc": None if np.isnan(auroc) else round(float(auroc), 4),
            **round_metric_dict(best_m),
        },
    }

    summary_path = OUTPUTS_DIR / f"eval_summary_{split}_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)
    print(f"  Per-image results CSV : {csv_path}")
    print(f"  Summary JSON          : {summary_path}")
    print()
    print("  The per-image CSV is your input to the segment aggregation pipeline.")
    print("  Columns: image_key, gt_label, na_confident, max_confidence, n_detections")
    print()

    # ── Optional: save prediction visualizations ──────────────────────────────
    if args.save_images:
        import random

        print("Saving sample prediction visualizations...")
        vis_dir = OUTPUTS_DIR / f"vis_{split}_{run_name}"
        vis_dir.mkdir(exist_ok=True)

        pos_keys = [r["image_key"] for r in per_image_rows if r["gt_label"] == 1]
        neg_keys = [r["image_key"] for r in per_image_rows if r["gt_label"] == 0]

        sample_keys = (
            random.sample(pos_keys, min(20, len(pos_keys))) +
            random.sample(neg_keys, min(20, len(neg_keys)))
        )
        sample_paths = [str(DATA_DIR / "images" / split / f"{k}.jpg") for k in sample_keys]

        model.predict(
            source=sample_paths,
            imgsz=args.imgsz,
            conf=best_t,
            iou=0.5,
            device=args.device if args.device else None,
            save=True,
            project=str(vis_dir),
            name="predictions",
            verbose=False,
        )
        print(f"  Visualizations saved to : {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained parking sign detector")
    parser.add_argument("--weights", required=True, help="Path to best.pt from training run")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--device", default="", help="'cpu', 'mps', '0', etc.")
    parser.add_argument("--chunk-size", dest="chunk_size", default=64, type=int,
                        help="Number of images per prediction chunk for image-level inference")
    parser.add_argument("--pred-conf", dest="pred_conf", default=0.1, type=float,
                        help="Low confidence threshold used before threshold sweep")
    parser.add_argument("--save-images", action="store_true",
                        help="Save sample prediction visualizations")
    args = parser.parse_args()
    run(args)