
import multiprocessing
import torch
from ultralytics import YOLO

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n-cls.pt")

    model.train(
        data="datasets/fire/prepared",
        epochs=30,
        imgsz=224,
        batch=32,
        device=0,
        workers=0,
        project="runs/fire",
        name="fire_v1",
        patience=8
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
