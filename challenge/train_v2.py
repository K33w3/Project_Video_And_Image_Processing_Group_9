import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="D:/playground/image-video-processing-course")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TEST_CSV = DATA_DIR / "test.csv"
OUT_CSV = DATA_DIR / "submission.csv"

NUM_CLASSES = 10
IMG_SIZE = 96
BATCH = args.batch
EPOCHS = args.epochs
LR = args.lr
SEED = args.seed
VAL_SPLIT = 0.1
MIXUP_ALPHA = 0.3
LABEL_SMOOTH = 0.1

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} epochs={EPOCHS} batch={BATCH}")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class TrainDS(Dataset):
    def __init__(self, root, tf=None):
        self.tf = tf
        self.images = []
        self.labels = []
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            lbl = int(cls_dir.name)
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() == ".png":
                    img = Image.open(p).convert("RGB")
                    img.load()
                    self.images.append(img)
                    self.labels.append(lbl)
        print(f"{len(self.images)} images loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.tf:
            img = self.tf(img)
        return img, self.labels[i]


class TestDS(Dataset):
    def __init__(self, csv_path, root, tf=None):
        self.tf = tf
        self.root = root
        self.ids = []
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                self.ids.append(int(row[0]))
        self.images = []
        for img_id in self.ids:
            img = Image.open(root / f"{img_id}.png").convert("RGB")
            img.load()
            self.images.append(img)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.images[i]
        if self.tf:
            img = self.tf(img)
        return img, self.ids[i]


class AugWrapper(Dataset):
    def __init__(self, base_ds, indices, tf):
        self.base = base_ds
        self.indices = indices
        self.tf = tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.base.images[self.indices[i]]
        lbl = self.base.labels[self.indices[i]]
        return self.tf(img), lbl


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return m.to(device)


def mixup(x, y, alpha=0.3):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return x * lam + x[idx] * (1 - lam), y, y[idx], lam


base_ds = TrainDS(TRAIN_DIR, tf=None)
n_val = int(len(base_ds) * VAL_SPLIT)
n_train = len(base_ds) - n_val
all_idx = list(range(len(base_ds)))
random.shuffle(all_idx)
train_idx = all_idx[:n_train]
val_idx = all_idx[n_train:]

train_ds = AugWrapper(base_ds, train_idx, train_tf)
val_ds = AugWrapper(base_ds, val_idx, eval_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

test_ds = TestDS(TEST_CSV, TEST_DIR, tf=eval_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(f"train={n_train} val={n_val} test={len(test_ds)}")

model = build_model()
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        xm, ya, yb, lam = mixup(x, y, MIXUP_ALPHA)
        opt.zero_grad()
        out = model(xm)
        loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    sched.step()

    model.eval()
    tot, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
    val_acc = correct / tot
    print(f"[{epoch}/{EPOCHS}] val={val_acc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), DATA_DIR / "best_model.pt")

model.load_state_dict(torch.load(DATA_DIR / "best_model.pt", map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(device)
        out = model(x)
        for img_id, cat in zip(ids.tolist(), out.argmax(1).tolist()):
            preds.append((img_id, cat))

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Id", "Category"])
    for img_id, cat in preds:
        w.writerow([img_id, cat])
print(f"done, {len(preds)} rows")
