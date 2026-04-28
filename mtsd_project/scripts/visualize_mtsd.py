import json
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtsd"
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"


def load_annotation(image_key):
    annotation_path = ANNOTATIONS_DIR / f"{image_key}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"No annotation found for image key: {image_key}")
    with open(annotation_path, "r") as f:
        return json.load(f)


def load_image(image_key):
    image_path = IMAGES_DIR / f"{image_key}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"No image found for image key: {image_key}")
    return Image.open(image_path).convert("RGB")


def get_font():
    try:
        return ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def get_random_image_key():
    annotation_files = list(ANNOTATIONS_DIR.glob("*.json"))
    if not annotation_files:
        raise RuntimeError("No annotations found in dataset/annotations")
    return random.choice(annotation_files).stem


def visualize_image(image_key, save_path=None):
    anno = load_annotation(image_key)
    img = load_image(image_key)
    draw = ImageDraw.Draw(img)
    font = get_font()

    for obj in anno.get("objects", []):
        bbox = obj["bbox"]
        x1 = bbox["xmin"]
        y1 = bbox["ymin"]
        x2 = bbox["xmax"]
        y2 = bbox["ymax"]
        label = obj["label"]

        if x1 <= x2:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 3, y1 + 3), label, fill="yellow", font=font)
        else:
            cross_boundary = bbox.get("cross_boundary")
            if cross_boundary:
                left = cross_boundary.get("left")
                right = cross_boundary.get("right")
                if left:
                    draw.rectangle(
                        [left["xmin"], left["ymin"], left["xmax"], left["ymax"]],
                        outline="blue",
                        width=3,
                    )
                    draw.text(
                        (left["xmin"] + 3, left["ymin"] + 3),
                        label,
                        fill="yellow",
                        font=font,
                    )
                if right:
                    draw.rectangle(
                        [right["xmin"], right["ymin"], right["xmax"], right["ymax"]],
                        outline="blue",
                        width=3,
                    )
                    draw.text(
                        (right["xmin"] + 3, right["ymin"] + 3),
                        label,
                        fill="yellow",
                        font=font,
                    )

    if save_path:
        img.save(save_path)
        print(f"Saved visualization to: {save_path}")
    else:
        img.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_key = sys.argv[1]
    else:
        image_key = get_random_image_key()

    print(f"Visualizing image_key: {image_key}")
    visualize_image(image_key)