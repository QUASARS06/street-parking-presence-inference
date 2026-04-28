#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def noisy_or(probs):
    return 1 - np.prod(1 - probs)


def compute_segment_features(df):
    rows = []

    for seg_id, group in tqdm(df.groupby("segment_id"), desc="Aggregating segments"):
        sign_scores = group["pred_sign_score"].values
        meter_scores = group["pred_meter_score"].values
        curb_scores = group["pred_curb_strong_color_score"].values

        row = {
            "segment_id": seg_id,
            "segment_label": group["segment_label"].iloc[0],
            "segment_type": group["segment_type"].iloc[0],
        }

        # SIGN
        row.update({
            "sign_max": sign_scores.max(),
            "sign_mean": sign_scores.mean(),
            "sign_count_015": (sign_scores >= 0.15).sum(),
            "sign_noisy_or": noisy_or(sign_scores),
        })

        # METER
        row.update({
            "meter_max": meter_scores.max(),
            "meter_mean": meter_scores.mean(),
            "meter_count_01": (meter_scores >= 0.1).sum(),
            "meter_noisy_or": noisy_or(meter_scores),
        })

        # CURB
        row.update({
            "curb_max": curb_scores.max(),
            "curb_mean": curb_scores.mean(),
            "curb_count_02": (curb_scores >= 0.2).sum(),
            "curb_noisy_or": noisy_or(curb_scores),
        })

        # COMBINED (simple but effective)
        row["combined_score"] = max(
            row["sign_max"],
            0.6 * row["meter_max"],
            0.4 * row["curb_max"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0

    return p, r, f1, auc


def run():
    root = Path.home() / "Downloads/parking_aggregation_project"
    input_csv = root / "outputs/synthetic_segments/synthetic_segment_image_predictions.csv"

    df = pd.read_csv(input_csv)

    print("Computing segment features...")
    seg_df = compute_segment_features(df)

    print("\nEvaluating aggregation...")

    thresholds = np.linspace(0.05, 0.95, 19)

    best = None

    for t in thresholds:
        p, r, f1, auc = evaluate(
            seg_df["segment_label"].values,
            seg_df["combined_score"].values,
            t
        )

        print(f"t={t:.2f} | P={p:.3f} R={r:.3f} F1={f1:.3f} AUC={auc:.3f}")

        if best is None or f1 > best["f1"]:
            best = {"t": t, "p": p, "r": r, "f1": f1, "auc": auc}

    print("\nBest threshold:")
    print(best)

    # ---- BASELINE (single image per segment) ----
    print("\nEvaluating SINGLE IMAGE (per-segment) baseline...")

    single_rows = []

    for seg_id, group in df.groupby("segment_id"):
        # Option 1: first image (deterministic)
        row = group.iloc[0]

        # Option 2 (better): random image
        # row = group.sample(1, random_state=42).iloc[0]

        single_rows.append({
            "segment_id": seg_id,
            "segment_label": row["segment_label"],
            "score": row["pred_sign_score"]  # strongest cue
        })

    single_df = pd.DataFrame(single_rows)

    for t in thresholds:
        p, r, f1, auc = evaluate(
            single_df["segment_label"].values,
            single_df["score"].values,
            t
        )
        print(f"[Single-Segment] t={t:.2f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Save
    seg_df.to_csv(root / "outputs/synthetic_segments/segment_level_features.csv", index=False)
    print("\nSaved segment-level features")


if __name__ == "__main__":
    run()
