
import multiprocessing
import torch
from ultralytics import YOLO


def main():
    print("AEROX RESCUE AI - Training")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check PyTorch installation.")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="runs/aerox",
        name="rescue_v1"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
