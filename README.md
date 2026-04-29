# Street Parking Presence Inference

CS 766 *(Computer Vision)* final project at the **University of Wisconsin–Madison**, Spring 2026.

**Authors:** Chirag Jain · Ritik Singh

We infer whether a **street segment** allows on-street parking from street-level imagery, by combining three image-level cue detectors — **parking signs**, **parking meters**, and **curb structure + color** — and aggregating their evidence across multiple nearby views. Single-image inference is brittle because parking cues are sparse and viewpoint-dependent; multi-view aggregation lifts F1 from **0.32** (single-image baseline) to **0.83** on a controlled synthetic benchmark, and recovers **5 of 6** segments on a manually-collected real-world dataset.

**Project website:** <https://quasars06.github.io/cs766-street-parking/>

**Watch presentation:** [Documents tab](https://quasars06.github.io/cs766-street-parking/documents.html) on the website (slides + video).

---

## Repository structure

```
street-parking-presence-inference/
├── mtsd_project/                # Parking-sign detection on MTSD
│   ├── scripts/                 #   data prep, training, evaluation, viz
│   ├── dataset/                 #   YOLO-format binary parking-sign dataset (5-image samples)
│   ├── data/                    #   processed images / labels (samples)
│   ├── outputs/                 #   eval CSVs, threshold sweeps, category samples
│   ├── sample_images/           #   manually collected qualitative images
│   └── parking_signs.yaml       #   YOLO data config
│
├── Mapillary_Vistas_Dataset/    # Curb segmentation + zero-shot meter eval
│   ├── scripts/                 #   curb U-Net training, meter zero-shot eval
│   ├── curb_segmentation/       #   mask metadata + samples
│   ├── training/                #   5-image dataset samples
│   ├── validation/              #
│   ├── testing/                 #
│   ├── parking_meter_imgs/      #   Vistas parking-meter samples
│   └── outputs/                 #   meter eval CSVs
│
├── manual_segments_dataset/     # 30-image real-world qualitative dataset
│   ├── segments/                #   6 segments × 5 views (raw images)
│   ├── outputs/                 #   visualization outputs + scores CSVs
│   └── segments.csv             #   index
│
└── parking_aggregation_project/ # Segment-level aggregation system
    ├── scripts/                 #   synthetic-segment builder, aggregation, eval
    ├── metadata/                #   per-image cue scores
    └── outputs/synthetic_segments/   # synthetic benchmark + per-threshold results
```

Each top-level folder is a self-contained sub-project with its own scripts and outputs. The full design rationale, results, and qualitative analysis are on the [project website](https://quasars06.github.io/cs766-street-parking/).

## Trained weights & manual-segments dataset

Distributed via the **[v1.0 release](https://github.com/QUASARS06/street-parking-presence-inference/releases/tag/v1.0)**:

| Asset | Description |
|---|---|
| `parking_sign_best.pt` | YOLOv8m parking-sign detector (50 epochs, mAP@50 = 0.5487, image-level F1 = 0.6673). |
| `curb_best.pt` | U-Net curb segmentation (best val Dice = 0.5184 at epoch 17). |
| `segments.zip` | 30 manually-collected real-world segment images (6 × 5) used in the qualitative aggregation experiments, plus visualization outputs. |

The model checkpoints are excluded from the git repo via `.gitignore` because of size and the GitHub 100 MB per-file limit.

## Reproducing the main results

1. Download the full datasets from their official sources — the repo only ships 5-image samples per split:
   - [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign) — for parking-sign training.
   - [Mapillary Vistas v2.0](https://www.mapillary.com/dataset/vistas) — for curb segmentation + parking-meter evaluation.
2. Build the binary parking-sign YOLO dataset and train / evaluate from `mtsd_project/scripts/`.
3. Train the curb segmentation U-Net and run the zero-shot parking-meter evaluation from `Mapillary_Vistas_Dataset/scripts/`.
4. Build the synthetic pseudo-segment benchmark and run the segment-level aggregation evaluation from `parking_aggregation_project/scripts/`.
5. Reproduce the annotated qualitative figures by running the visualization scripts against the `manual_segments_dataset/` bundle (also available as `segments.zip` in the v1.0 release).

## Environment

Experiments were primarily run on Kaggle (Tesla T4 / P100) and a local Apple Silicon machine.

- Python 3.10+
- PyTorch 2.x
- `ultralytics` (YOLOv8 / YOLO11)
- `segmentation-models-pytorch` (curb U-Net)
- `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`

Each sub-project has its own `requirements.txt`.

## Citation

If you find this work useful, please cite the project page:

```
Chirag Jain and Ritik Singh.
Street Parking Presence Inference from Street-Level Imagery
via Multi-Cue Detection and Geo-Aggregation.
CS 766 Computer Vision, University of Wisconsin–Madison, Spring 2026.
https://quasars06.github.io/cs766-street-parking/
```

---

*Issues, questions, or reproductions welcome — open an issue on this repo.*
