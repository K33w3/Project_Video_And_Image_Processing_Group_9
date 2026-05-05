import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

DATA_DIR = Path("D:/playground/image-video-processing-course")
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
TEST_CSV = DATA_DIR / "test.csv"
OUT_CSV = DATA_DIR / "submission.csv"

NUM_CLASSES = 10
IMG_SIZE = 96
BATCH = 64
EPOCHS = 15
LR = 1e-3
SEED = 42
VAL_SPLIT = 0.1

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# use ImageNet stats since we're using a pretrained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class TrainDS(Dataset):
    def __init__(self, root, tf=None):
        self.tf = tf
        self.samples = []
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            lbl = int(cls_dir.name)
            for p in cls_dir.iterdir():
                if p.suffix.lower() == ".png":
                    self.samples.append((p, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.tf:
            img = self.tf(img)
        return img, lbl


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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(self.root / f"{img_id}.png").convert("RGB")
        if self.tf:
            img = self.tf(img)
        return img, img_id


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return m.to(device)


full_ds = TrainDS(TRAIN_DIR, tf=train_tf)
n_val = int(len(full_ds) * VAL_SPLIT)
n_train = len(full_ds) - n_val
train_set, val_set_raw = random_split(full_ds, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(SEED))

eval_ds = TrainDS(TRAIN_DIR, tf=eval_tf)
import torch.utils.data
val_set = torch.utils.data.Subset(eval_ds, val_set_raw.indices)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=0)

test_ds = TestDS(TEST_CSV, TEST_DIR, tf=eval_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(f"train={n_train} val={n_val} test={len(test_ds)}")

model = build_model()
loss_fn = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    tr_acc = correct / tot

    model.eval()
    tot, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
    val_acc = correct / tot
    sched.step()
    print(f"[{epoch}/{EPOCHS}] train={tr_acc:.4f} val={val_acc:.4f}")
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
print(f"done, wrote {OUT_CSV}")
