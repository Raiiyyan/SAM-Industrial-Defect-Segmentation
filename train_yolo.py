from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Defect Training...")

    model = YOLO("yolov8m.pt")  # Upgrade from nano

    results = model.train(
        data=r"G:\CSE465\data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,                  # Safe with 16GB VRAM on yolov8m
        device="0",
        name="universal_defect_v1",
        patience=20,               # Early stopping
        freeze=10,                 # Freeze first 10 backbone layers
        augment=True,              # Mosaic, flips, HSV jitter, etc.
        val=True,                  # Validate every epoch
        save_period=10,            # Checkpoint every 10 epochs
        cos_lr=True,               # Cosine LR schedule, smoother convergence
        label_smoothing=0.1,       # Helps with noisy defect labels
        workers=8,
    )

    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)')}")
    print("Weights saved to runs/detect/universal_defect_v1/weights/best.pt")

if __name__ == '__main__':
    main()