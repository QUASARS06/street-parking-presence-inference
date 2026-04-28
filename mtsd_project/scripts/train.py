"""
train.py
────────
Fine-tunes YOLOv8m on the prepared parking sign dataset.

BATCH TRAINING WORKFLOW (recommended for Kaggle):
    Run in batches of ~10 epochs at a time. Each batch resumes from the
    last checkpoint and runs until --epochs is reached.

    Batch 1 (fresh start, epochs 1-10):
        python scripts/train.py --device 0 --batch 32 --epochs 10 --cls 2.0

    Batch 2 (resume, epochs 11-20):
        python scripts/train.py --device 0 --batch 32 --epochs 20 --cls 2.0 \
            --model checkpoints/last.pt --resume

    Batch 3 (resume, epochs 21-30):
        python scripts/train.py --device 0 --batch 32 --epochs 30 --cls 2.0 \
            --model checkpoints/last.pt --resume

    ... and so on until --epochs 50

    IMPORTANT: After each batch, download last.pt from the Kaggle Output tab
    and upload it as checkpoints/last.pt for the next batch.

OTHER USAGE:
    python scripts/train.py --device mps                  # Mac M1 fresh start
    python scripts/train.py --fraction 0.1 --epochs 5    # smoke test
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
YAML_PATH     = PROJECT_ROOT / "parking_signs.yaml"
RUNS_DIR      = PROJECT_ROOT / "runs"
KAGGLE_OUTPUT = Path("/kaggle/working/runs")

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULTS = dict(
    model         = "yolov8m.pt",
    imgsz         = 640,
    epochs        = 50,
    batch         = 16,
    lr0           = 0.01,
    lrf           = 0.01,
    warmup_epochs = 3,
    patience      = 100,      # set high — we manage stopping manually via batches
    workers       = 4,
    device        = "",
    seed          = 42,
    cls           = 0.5,
    fraction      = 1.0,
)


# ── Auto-save callback ────────────────────────────────────────────────────────

def save_to_output(run_dir):
    """Copy run folder to /kaggle/working/runs so it appears in Kaggle Output tab."""
    try:
        KAGGLE_OUTPUT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            run_dir,
            KAGGLE_OUTPUT / run_dir.name,
            dirs_exist_ok=True
        )
        print(f"  Saved to output: {KAGGLE_OUTPUT / run_dir.name}")
    except Exception as e:
        print(f"  Save failed (non-fatal): {e}")


class SaveCallback:
    """Saves checkpoint to /kaggle/working/runs every 3 epochs."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        if epoch % 3 == 0:
            print(f"\n  Auto-saving at epoch {epoch}...")
            save_to_output(self.run_dir)


# ── Training ──────────────────────────────────────────────────────────────────

def run(args):
    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {YAML_PATH}\n"
            "Run prepare_dataset.py first, or write the YAML manually."
        )

    # Fresh start: generate timestamped run name
    # Resume: YOLO derives run name from the checkpoint automatically
    if args.resume:
        run_name = None
    else:
        run_name = (
            f"parking_sign_{args.model.replace('.pt', '')}"
            f"_imgsz{args.imgsz}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    print("=" * 60)
    print("PARKING SIGN DETECTOR - TRAINING")
    print("=" * 60)
    print(f"  Model         : {args.model}")
    print(f"  Resume        : {args.resume}")
    print(f"  Target epochs : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Image size    : {args.imgsz}")
    print(f"  Early stop    : {args.patience} epochs")
    if run_name:
        print(f"  Run name      : {run_name}")
    print()

    if args.resume:
        checkpoint_path = Path(args.model)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Make sure you uploaded last.pt to the checkpoints/ folder."
            )
        # Run dir is two levels up from weights/last.pt
        run_dir = checkpoint_path.parent.parent
        print(f"  Resuming from : {checkpoint_path}")
        print(f"  Run dir       : {run_dir}")
    else:
        run_dir = RUNS_DIR / "detect" / run_name

    print()

    model = YOLO(args.model)

    cb = SaveCallback(run_dir)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)

    train_kwargs = dict(
        data          = str(YAML_PATH),
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        lr0           = args.lr0,
        lrf           = args.lrf,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        workers       = args.workers,
        device        = args.device if args.device else None,
        seed          = args.seed,
        mosaic        = 1.0,
        mixup         = 0.1,
        copy_paste    = 0.1,
        cls           = args.cls,
        fraction      = args.fraction,
        project       = str(RUNS_DIR / "detect"),
        exist_ok      = args.exist_ok,
        pretrained    = not args.scratch,
        resume        = args.resume,
        val           = True,
        save          = True,
        save_period   = 10,
        plots         = True,
        verbose       = True,
    )

    # Only pass name for fresh runs — resume derives name from checkpoint
    if run_name:
        train_kwargs["name"] = run_name

    results = model.train(**train_kwargs)

    # Final save to Kaggle output
    print("\nSaving final results to output tab...")
    save_to_output(run_dir)

    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Best weights  : {best_weights}")
    print(f"  Last weights  : {last_weights}")
    print(f"    mAP50       : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"    mAP50-95    : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"    Precision   : {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"    Recall      : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print()
    print("  NEXT BATCH: Download last.pt from Output tab, upload to")
    print("  checkpoints/last.pt, then run with --resume --epochs N+10")

    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 parking sign detector")
    parser.add_argument("--model",         default=DEFAULTS["model"],
                        help="Model variant (yolov8m.pt) or path to checkpoint (checkpoints/last.pt)")
    parser.add_argument("--imgsz",         default=DEFAULTS["imgsz"],         type=int)
    parser.add_argument("--epochs",        default=DEFAULTS["epochs"],        type=int,
                        help="Target epoch number. For batches: set to current_epoch + batch_size")
    parser.add_argument("--batch",         default=DEFAULTS["batch"],         type=int)
    parser.add_argument("--lr0",           default=DEFAULTS["lr0"],           type=float)
    parser.add_argument("--lrf",           default=DEFAULTS["lrf"],           type=float)
    parser.add_argument("--warmup-epochs", default=DEFAULTS["warmup_epochs"], type=int)
    parser.add_argument("--patience",      default=DEFAULTS["patience"],      type=int,
                        help="Early stopping patience. Set high (100) for batch training.")
    parser.add_argument("--workers",       default=DEFAULTS["workers"],       type=int)
    parser.add_argument("--device",        default=DEFAULTS["device"],
                        help="Device: 'cpu', 'mps' (Mac M1), '0' (GPU)")
    parser.add_argument("--seed",          default=DEFAULTS["seed"],          type=int)
    parser.add_argument("--cls",           default=DEFAULTS["cls"],           type=float,
                        help="Classification loss weight. Use 2.0 for class imbalance.")
    parser.add_argument("--fraction",      default=DEFAULTS["fraction"],      type=float,
                        help="Dataset fraction. Use 0.1 for smoke test.")
    parser.add_argument("--cls-pw",        default=1.0,                       type=float,
                        help="Ignored — removed in newer Ultralytics versions.")
    parser.add_argument("--scratch",       action="store_true",
                        help="Train from scratch, no pretrained weights.")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from checkpoint passed to --model.")
    parser.add_argument("--exist-ok",      action="store_true",
                        help="Allow overwriting existing run directory.")
    args = parser.parse_args()
    run(args)