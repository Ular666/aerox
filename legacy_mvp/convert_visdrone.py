
from pathlib import Path
from PIL import Image
import random
import shutil

SOURCE = Path(
    "datasets/VisDrone2019-DET-train/"
    "VisDrone2019-DET-train"
)

OUTPUT = Path("datasets/aerox_rescue")

random.seed(42)

images = sorted(
    p for p in (SOURCE / "images").iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

random.shuffle(images)

# Development split from the training dataset.
# An official independent validation set is still needed.
split = int(len(images) * 0.8)

for index, image_path in enumerate(images):
    subset = "train" if index < split else "val"

    image_dir = OUTPUT / "images" / subset
    label_dir = OUTPUT / "labels" / subset

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size

    annotation = SOURCE / "annotations" / (
        image_path.stem + ".txt"
    )

    labels = []

    if annotation.exists():
        for line in annotation.read_text().splitlines():
            fields = line.strip().split(",")

            if len(fields) < 8:
                continue

            x, y, w, h = map(float, fields[:4])
            score = int(fields[4])
            category = int(fields[5])

            # Keep valid pedestrian and people annotations.
            if category not in [1, 2] or score != 1:
                continue

            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            nw = (x2 - x1) / width
            nh = (y2 - y1) / height

            labels.append(
                f"0 {xc:.6f} {yc:.6f} "
                f"{nw:.6f} {nh:.6f}"
            )

    shutil.copy2(image_path, image_dir / image_path.name)

    (label_dir / (image_path.stem + ".txt")).write_text(
        "\n".join(labels)
    )

    if (index + 1) % 500 == 0:
        print(f"Processed {index + 1}/{len(images)}")

print("Dataset conversion completed!")
print(f"Train: {split}")
print(f"Validation: {len(images) - split}")

