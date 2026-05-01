# IIVP 2026 Challenge — Solution

**Competition:** [IIVP 2026 Challenge](https://www.kaggle.com/competitions/iivp-2026-challenge)
**Course:** KEN3238 Image and Video Processing
**Public leaderboard score:** 1.0000000 (100%)

---

## Problem

10-class image classification of 32×32 grayscale handwritten character/symbol images.

- **Train set:** 17,000 images (1,700 per class, classes 0–9) in `train/train/{class_id}/*.png`
- **Test set:** 3,000 images in `test/test/*.png`
- **Submission format:** `Id,Category` CSV

The images are handwritten symbols on a black background — visually similar characters (e.g. 0 vs O, 1 vs I) make this non-trivial even for humans.

---

## Solution Overview

Pretrained **EfficientNet-B0** fine-tuned end-to-end, with a 3-seed ensemble and 5-pass test-time augmentation (TTA).

### Why EfficientNet-B0?

- Pretrained ImageNet weights provide strong low-level feature detectors (edges, curves, strokes) that transfer directly to handwritten characters
- Converges extremely fast: 98%+ validation accuracy by epoch 2
- Small enough to fit 3 full training runs on an 8 GB laptop GPU in ~40 minutes

### Architecture

```
Input (32×32 grayscale)
  → convert to 3-channel RGB (channel replication)
  → resize to 96×96
  → EfficientNet-B0 backbone (ImageNet pretrained)
  → Dropout(0.3)
  → Linear(1280 → 10)
  → softmax
```

### Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 peak, cosine decay |
| LR warmup | 3 epochs (linear ramp) |
| Epochs | 40 per seed |
| Batch size | 256 |
| Precision | Mixed (AMP, float16) |
| Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB) |
| Time | ~14 min/seed → ~42 min total |

### Augmentation

Applied per-batch during training only (no augmentation at inference except TTA):

- `RandomRotation(±15°)`
- `RandomAffine(degrees=0, translate=0.1, scale=0.9–1.1)`
- `RandomPerspective(distortion=0.2)`
- `RandomErasing(p=0.1)`
- `Mixup` (α=0.3) — interpolates pairs of images and their labels
- `Label smoothing` (ε=0.1) — prevents overconfident predictions

No horizontal flip — the symbols are not left-right symmetric, so flipping would corrupt the labels.

### Ensemble + TTA

**3-seed ensemble:** Train identical models with seeds 42, 123, and 456. Randomness in weight init, batch shuffling, and augmentation ensures diversity. Final prediction = average of softmax probabilities across all three models.

**5-pass TTA:** For each test image, run inference on:
1. Original image
2. Rotated +8°
3. Rotated −8°
4. Rotated +15°
5. Rotated −15°

Average the 5 softmax outputs before combining with the ensemble. Total inference runs per test image: 3 seeds × 5 TTA passes = **15 forward passes**.

---

## Results

### Validation accuracy per seed (10% holdout, stratified)

| Seed | Best val accuracy |
|------|------------------|
| 42   | 99.94%           |
| 123  | 100.00%          |
| 456  | 100.00%          |

### Kaggle leaderboard

| Split | Score |
|-------|-------|
| Public | **1.0000000** |

---

## Reproducing the Solution

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training (strongly recommended — CPU is ~10× slower):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Download the dataset

```bash
export KAGGLE_API_TOKEN="<your token>"
kaggle competitions download -c iivp-2026-challenge
python -c "import zipfile; zipfile.ZipFile('iivp-2026-challenge.zip').extractall('.')"
```

This produces:
```
train/train/{0-9}/*.png   # training images
test/test/*.png           # test images
sample_submission.csv
```

### 3. Train and generate submission

```bash
python train_v2.py --data-dir /path/to/dataset
```

Default flags: `--epochs 40 --seeds 42 123 456 --tta-n 5 --batch 256 --img-size 96`

Outputs `submission.csv` in the dataset directory when complete.

Full option reference:

```
--data-dir    Path to dataset root (default: IIVP_DATA_DIR env var or parent of script)
--epochs      Epochs per seed (default: 40)
--seeds       Random seeds for ensemble (default: 42 123 456)
--tta-n       TTA passes at inference (default: 5)
--batch       Batch size (default: 256, reduce if OOM)
--img-size    Resize target in pixels (default: 96)
--lr          Peak learning rate (default: 5e-4)
--workers     DataLoader worker count (default: 0)
```

---

## Files

| File | Description |
|------|-------------|
| `train_v2.py` | Main solution — EfficientNet-B0 ensemble with TTA |
| `train_classifier.py` | Baseline compact CNN (reference only, not used for submission) |
| `submission.csv` | Final predictions submitted to Kaggle |
| `requirements.txt` | Python dependencies |

---

## Key Design Decisions

**RAM preloading:** All 17,000 training images are decoded and stored as PIL objects in RAM at startup. This avoids repeated disk I/O across 40 epochs × 3 seeds = 120 total passes over the dataset.

**No frozen backbone:** All EfficientNet-B0 layers are fine-tuned. With only 40 epochs and a warmup schedule, catastrophic forgetting is not a concern, and full fine-tuning gives better final accuracy.

**Image size 96×96 over 64×64:** Slightly larger input preserves more spatial detail for the pretrained backbone's early conv layers, measurably improving epoch-1 accuracy (85% vs 70%) and reducing the number of epochs needed to converge.
