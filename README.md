# IIVP 2026 Challenge — Group 9

10-class handwritten character classification on 32×32 grayscale images.
Kaggle public score: **1.0000000**

## Approach

EfficientNet-B0 pretrained on ImageNet, fine-tuned end-to-end on the challenge dataset. The grayscale images are replicated to 3 channels and upscaled to 96×96 to make better use of the pretrained weights.

**Training:**
- Optimizer: AdamW (weight decay 1e-4)
- Learning rate: 5e-4 with cosine annealing and 3-epoch linear warmup
- Label smoothing ε = 0.1, Mixup α = 0.3
- Augmentation: random rotation ±15°, affine, perspective, random erasing
- 40 epochs per seed, AMP (float16) on CUDA

**Inference:**
- 3 models trained with different seeds (42, 123, 456)
- 5-pass test-time augmentation per model (original + 4 slight rotations)
- Final prediction = argmax of averaged softmax probabilities (15 total passes per image)

## Setup

```
pip install -r requirements.txt
```

GPU recommended. For CUDA 12.8:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```
python train_v2.py --data-dir /path/to/iivp-2026-challenge
```

The dataset directory should contain `train/train/{0-9}/` and `test/test/`, plus `test.csv`. The script writes `submission.csv` to the same directory when done.

Available options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | script parent | Path to dataset root |
| `--epochs` | 40 | Epochs per seed |
| `--seeds` | 42 123 456 | Seeds for ensemble |
| `--tta-n` | 5 | TTA passes at inference |
| `--batch` | 256 | Batch size |
| `--img-size` | 96 | Input resolution |
| `--lr` | 5e-4 | Peak learning rate |


## Additional Experimental Methods

Besides the final EfficientNet-B0 solution, additional methods were tried during development.

### KNN + HOG

A traditional computer vision pipeline that uses:
- image recentering
- HOG (Histogram of Oriented Gradients) feature extraction
- K-Nearest Neighbors classification

Method Pipeline:
1. Load grayscale digit images.
2. Recenter digits using center-of-mass alignment.
3. Extract HOG descriptors.
4. Train KNN classifier (`k=3`). 

This approach resulted in approximately `99.56%` validation accuracy.

### LeNet-5 CNN

LeNet-5-style convolutional neural network implemented in PyTorch.

Architecture:
- Conv(1->6, 5x5)
- MaxPool
- Conv(6->16, 5x5)
- MaxPool
- Fully connected layers:
  - 400 -> 120
  - 120 -> 84
  - 84 -> 10

Training:
- Adam optimizer
- Cross-entropy loss
- 35 epochs
- 80/20 train/validation 

This approach resulted in approximately `99.85%` validation accuracy.

## Results

| Seed | Best validation accuracy |
|------|--------------------------|
| 42   | 99.94%                   |
| 123  | 100.00%                  |
| 456  | 100.00%                  |

Kaggle public leaderboard: **1.0000000**
