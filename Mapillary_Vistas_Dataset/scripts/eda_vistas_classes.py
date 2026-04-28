import json
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# -------- PATH --------
DATASET_ROOT = Path(".")  # change this
POLYGON_DIR = DATASET_ROOT / "training" / "v2.0" / "polygons"

# -------- KEYWORDS --------
PARKING_KEYWORDS = [
    "parking",
    "meter",
    "pay",
    "ticket",
    "curb",
    "kerb",
    "sign",
    "pole"
]

def main():
    json_files = list(POLYGON_DIR.glob("*.json"))

    print(f"Total JSON files: {len(json_files)}")

    label_counter = Counter()
    category_counter = Counter()
    hierarchy = defaultdict(set)

    for jf in tqdm(json_files):
        with open(jf) as f:
            data = json.load(f)

        for obj in data["objects"]:
            label = obj["label"]
            label_counter[label] += 1

            parts = label.split("--")
            if len(parts) >= 1:
                category_counter[parts[0]] += 1

            # store hierarchy
            if len(parts) >= 2:
                hierarchy[parts[0]].add(parts[1])

    # -------- PRINT SUMMARY --------
    print("\n" + "="*60)
    print("TOTAL UNIQUE CLASSES")
    print("="*60)
    print(len(label_counter))

    print("\nTOP 123 CLASSES")
    for label, count in label_counter.most_common(123):
        print(f"{label:50s} {count}")

    print("\n" + "="*60)
    print("TOP-LEVEL CATEGORIES")
    print("="*60)
    for cat, count in category_counter.most_common():
        print(f"{cat:20s} {count}")

    print("\n" + "="*60)
    print("CATEGORY HIERARCHY")
    print("="*60)
    for cat, subs in hierarchy.items():
        print(f"\n{cat}")
        for s in sorted(subs):
            print(f"  ├── {s}")

    # -------- PARKING RELATED --------
    print("\n" + "="*60)
    print("PARKING-RELATED CANDIDATES")
    print("="*60)

    parking_candidates = []

    for label, count in label_counter.items():
        if any(k in label.lower() for k in PARKING_KEYWORDS):
            parking_candidates.append((label, count))

    parking_candidates.sort(key=lambda x: -x[1])

    for label, count in parking_candidates:
        print(f"{label:50s} {count}")

    print(f"\nTotal parking-related candidates: {len(parking_candidates)}")

    # -------- SAVE --------
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "vistas_all_classes.txt", "w") as f:
        for label, count in label_counter.most_common():
            f.write(f"{label}\t{count}\n")

    with open(OUTPUT_DIR / "vistas_parking_candidates.txt", "w") as f:
        for label, count in parking_candidates:
            f.write(f"{label}\t{count}\n")

    print("\nSaved outputs to /outputs")

if __name__ == "__main__":
    main()