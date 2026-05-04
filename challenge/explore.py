from pathlib import Path
from collections import Counter
from PIL import Image

DATA_DIR = Path("D:/playground/image-video-processing-course")
TRAIN_DIR = DATA_DIR / "train" / "train"

classes = sorted(TRAIN_DIR.iterdir())
print(f"classes: {[c.name for c in classes]}")

counts = Counter()
for cls_dir in classes:
    imgs = list(cls_dir.glob("*.png"))
    counts[cls_dir.name] = len(imgs)

print(f"samples per class: {dict(counts)}")
print(f"total: {sum(counts.values())}")

sample = Image.open(next((classes[0]).glob("*.png")))
print(f"image size: {sample.size}, mode: {sample.mode}")
