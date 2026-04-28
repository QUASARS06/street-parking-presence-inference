import json
import csv
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

import gc


TARGET_LABEL = "object--parking-meter"
COCO_PARKING_METER_CLASS = 12  # COCO class index for parking meter


def batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def polygon_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()

    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            pairs.append((iou_xyxy(pb, gb), pi, gi))

    pairs.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for iou, pi, gi in pairs:
        if iou < iou_thresh:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_prf(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_dataset_items(dataset_root, split, positives_only=False, max_images=None):
    image_dir = dataset_root / split / "images"
    polygon_dir = dataset_root / split / "v2.0" / "polygons"

    json_files = sorted(polygon_dir.glob("*.json"))
    items = []

    for jf in tqdm(json_files, desc=f"Loading {split} annotations"):
        with open(jf, "r") as f:
            data = json.load(f)

        gt_boxes = []
        for obj in data.get("objects", []):
            if obj.get("label") != TARGET_LABEL:
                continue
            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue
            gt_boxes.append(polygon_to_bbox(polygon))

        if positives_only and not gt_boxes:
            continue

        image_path = image_dir / f"{jf.stem}.jpg"
        if not image_path.exists():
            continue

        items.append({
            "key": jf.stem,
            "image_path": str(image_path),
            "gt_boxes": gt_boxes,
            "gt_present": 1 if len(gt_boxes) > 0 else 0,
        })

        if max_images is not None and len(items) >= max_images:
            break

    return items


def run_eval(model, items, imgsz, conf, iou_match=0.5, device="", pred_batch=32):
    # image-level
    img_tp = img_fp = img_tn = img_fn = 0

    # box-level
    box_tp = box_fp = box_fn = 0

    num_batches = (len(items) + pred_batch - 1) // pred_batch

    for item_batch in tqdm(
        batched(items, pred_batch),
        desc=f"Batches imgsz={imgsz} conf={conf}",
        total=num_batches,
    ):
        image_paths = [it["image_path"] for it in item_batch]

        results = model.predict(
            source=image_paths,
            classes=[COCO_PARKING_METER_CLASS],
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            stream=True,
            device=device if device else None,
            max_det=20,
            batch=min(4, len(image_paths)),
        )

        for item, result in zip(item_batch, results):
            gt_boxes = item["gt_boxes"]

            pred_boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                pred_boxes = [
                    [float(x1), float(y1), float(x2), float(y2)]
                    for x1, y1, x2, y2 in xyxy
                ]

            gt_present = 1 if len(gt_boxes) > 0 else 0
            pred_present = 1 if len(pred_boxes) > 0 else 0

            if gt_present == 1 and pred_present == 1:
                img_tp += 1
            elif gt_present == 0 and pred_present == 1:
                img_fp += 1
            elif gt_present == 0 and pred_present == 0:
                img_tn += 1
            elif gt_present == 1 and pred_present == 0:
                img_fn += 1

            tp, fp, fn = match_boxes(gt_boxes, pred_boxes, iou_thresh=iou_match)
            box_tp += tp
            box_fp += fp
            box_fn += fn

        del results
        gc.collect()

    img_precision, img_recall, img_f1 = compute_prf(img_tp, img_fp, img_fn)
    img_accuracy = safe_div(img_tp + img_tn, img_tp + img_fp + img_tn + img_fn)

    box_precision, box_recall, box_f1 = compute_prf(box_tp, box_fp, box_fn)

    return {
        "imgsz": imgsz,
        "conf": conf,
        "iou_match": iou_match,
        "n_images": len(items),
        "n_positive_images": sum(it["gt_present"] for it in items),
        "n_negative_images": len(items) - sum(it["gt_present"] for it in items),
        "image_tp": img_tp,
        "image_fp": img_fp,
        "image_tn": img_tn,
        "image_fn": img_fn,
        "image_precision": img_precision,
        "image_recall": img_recall,
        "image_f1": img_f1,
        "image_accuracy": img_accuracy,
        "box_tp": box_tp,
        "box_fp": box_fp,
        "box_fn": box_fn,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": box_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Path to Mapillary Vistas root")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--imgsz-list", default="640,960,1280")
    parser.add_argument("--conf-list", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--iou-match", default=0.5, type=float)
    parser.add_argument("--positives-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--pred-batch", type=int, default=32)
    parser.add_argument("--device", default="")
    parser.add_argument("--out-csv", default="outputs/vistas_meter_eval.csv")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    imgsz_list = [int(x.strip()) for x in args.imgsz_list.split(",") if x.strip()]
    conf_list = [float(x.strip()) for x in args.conf_list.split(",") if x.strip()]

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    items = load_dataset_items(
        dataset_root=dataset_root,
        split=args.split,
        positives_only=args.positives_only,
        max_images=args.max_images,
    )

    print(f"Loaded items: {len(items)}")
    print(f"Positive images: {sum(it['gt_present'] for it in items)}")
    print(f"Negative images: {len(items) - sum(it['gt_present'] for it in items)}")

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    model = YOLO(args.model)

    rows = []
    for imgsz in imgsz_list:
        for conf in conf_list:
            print("=" * 70)
            print(f"EVAL: imgsz={imgsz}, conf={conf}")
            print("=" * 70)

            metrics = run_eval(
                model=model,
                items=items,
                imgsz=imgsz,
                conf=conf,
                iou_match=args.iou_match,
                device=args.device,
                pred_batch=args.pred_batch,
            )
            rows.append(metrics)

            gc.collect()

            print(
                f"[image-level] P={metrics['image_precision']:.4f} "
                f"R={metrics['image_recall']:.4f} "
                f"F1={metrics['image_f1']:.4f} "
                f"Acc={metrics['image_accuracy']:.4f}"
            )
            print(
                f"[box-level]   P={metrics['box_precision']:.4f} "
                f"R={metrics['box_recall']:.4f} "
                f"F1={metrics['box_f1']:.4f}"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print(f"Saved results to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()