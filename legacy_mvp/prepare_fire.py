
from pathlib import Path
from collections import defaultdict
import random
import shutil

ROOT = Path("datasets/fire")
SOURCE = next(p for p in ROOT.iterdir() if p.is_dir())
OUTPUT = ROOT / "prepared"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RNG = random.Random(42)

for split in ("train", "val", "test"):
    for cls in ("fire", "non fire", "Smoke"):
        (OUTPUT / split / cls).mkdir(parents=True, exist_ok=True)

for cls in ("fire", "non fire", "Smoke"):
    train_files = [
        p for p in (SOURCE / "train" / cls).iterdir()
        if p.suffix.lower() in EXTENSIONS
    ]
    test_files = [
        p for p in (SOURCE / "test" / cls).iterdir()
        if p.suffix.lower() in EXTENSIONS
    ]

    RNG.shuffle(train_files)
    val_count = int(len(train_files) * 0.10)

    splits = {
        "train": train_files[val_count:],
        "val": train_files[:val_count],
        "test": test_files,
    }

    for split, files in splits.items():
        destination = OUTPUT / split / cls
        for src in files:
            shutil.copy2(src, destination / src.name)

        print(f"{split:5} | {cls:8} | {len(files)}")

print("\nDataset prepared:", OUTPUT.resolve())
