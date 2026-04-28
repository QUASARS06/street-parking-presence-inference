from ultralytics import YOLO

model = YOLO("yolo11x.pt")   # COCO-pretrained
results = model.predict(
    source="parking_meter_imgs/park_meter_madison.jpg",
    classes=[12],            # COCO: parking meter
    conf=0.3,
    imgsz=1280,
    save=True,
    project="outputs/meter_zeroshot"
)