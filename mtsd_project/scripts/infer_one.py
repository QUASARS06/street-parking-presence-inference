from ultralytics import YOLO
from pathlib import Path

# IMAGE_PATH = "sample_images/parking_infer_sample.jpg"
# img_2_casa
# img_3_casa.JPG
# img_4_park_lot
# img_5_park_lot
# img_6_state_st
# img_7_park_st
IMAGE_PATH = "sample_images/madison/img_6_state_st.jpg"
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    imgsz=160,
    conf=0.15,
    iou=0.5,
    save=True,
    show=False,
    verbose=False,
)

for r in results:
    print(f"\nImage: {r.path}")
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        print("No parking sign detected.")
    else:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0]
            conf = float(b.conf.cpu().numpy()[0])
            print(f"Box {i+1}: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={conf:.4f}")

print("\nSaved output image is in runs/detect/predict/")