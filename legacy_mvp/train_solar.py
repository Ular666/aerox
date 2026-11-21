
import multiprocessing
import torch
from ultralytics import YOLO

def main():
    print("AEROX SOLAR AI - Training")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n-cls.pt")

    model.train(
        data="datasets/solar",
        epochs=30,
        imgsz=224,
        batch=32,
        device=0,
        workers=0,
        project="runs/solar",
        name="solar_v1"
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
