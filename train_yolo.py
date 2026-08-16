from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Defect Training on 80% Train / 20% Test Split...")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=r"G:\CSE465\data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        device="0",
        name="universal_defect_v1",
        patience=20,
        freeze=10,
        augment=True,
        val=True,
        save_period=10,
        cos_lr=True,
        label_smoothing=0.1,
        workers=8,
    )

    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)')}")
    print("Weights saved to runs/detect/universal_defect_v1/weights/best.pt")

if __name__ == '__main__':
    main()