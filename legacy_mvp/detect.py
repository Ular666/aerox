
from ultralytics import YOLO
import torch

print("=== AEROX AI CORE v0.1 ===")

device = 0 if torch.cuda.is_available() else "cpu"
print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")

# Load a pretrained object detection model
model = YOLO("yolo11n.pt")

# Run detection on a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=device,
    conf=0.25,
    save=True
)

for result in results:
    print("Detected objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"- {model.names[class_id]}: {confidence:.2%}")

print("Detection completed!")
