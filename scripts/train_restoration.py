"""
Train a lightweight U-Net for image restoration (denoising/deblurring/super-res).

The model learns to restore corrupted images to their clean originals.
Training pairs are generated on-the-fly using the same corruption functions
as the test set generation (build_corrupted_testsets.py).

Usage:
    python -m scripts.train_restoration
"""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import random
import time
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from scripts.restoration_net import RestorationUNet
from scripts.augmentations import apply_noise, apply_motion_blur, apply_lowres
from scripts.augmentations import NOISE_SIGMA, BLUR_KERNEL, BLUR_ANGLE_DEG, DOWNSCALE_FACTOR

# ====== Hyperparameters ======
SEED = 42
EPOCHS = 60
BATCH_SIZE = 8
PATCH_SIZE = 256
LR = 1e-3
NUM_WORKERS = 0

DATA_ROOT = Path("data/processed/visdrone_coco6")
TRAIN_IMG_DIR = DATA_ROOT / "images" / "train"
VAL_IMG_DIR = DATA_ROOT / "images" / "val"

OUT_DIR = Path("experiments/restoration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────
# Dataset: on-the-fly corruption pairs
# ──────────────────────────────────────────────
class RestorationDataset(Dataset):
    """
    Returns (corrupted_patch, clean_patch) pairs.
    Corruption is applied randomly each time (noise/blur/lowres).
    """

    def __init__(self, img_dir: Path, patch_size: int = PATCH_SIZE, is_train: bool = True):
        self.img_paths = sorted(img_dir.glob("*.jpg"))
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def _random_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h < size or w < size:
            img = cv2.resize(img, (max(w, size), max(h, size)))
            h, w = img.shape[:2]
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return img[y:y + size, x:x + size]

    def _center_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h < size or w < size:
            img = cv2.resize(img, (max(w, size), max(h, size)))
            h, w = img.shape[:2]
        y = (h - size) // 2
        x = (w - size) // 2
        return img[y:y + size, x:x + size]

    def _apply_random_corruption(self, img_bgr: np.ndarray) -> np.ndarray:
        choice = random.choice(["noise", "blur", "lowres"])
        if choice == "noise":
            return apply_noise(img_bgr, NOISE_SIGMA)
        elif choice == "blur":
            return apply_motion_blur(img_bgr, BLUR_KERNEL, BLUR_ANGLE_DEG)
        else:
            return apply_lowres(img_bgr, DOWNSCALE_FACTOR)

    def __getitem__(self, idx: int):
        img_bgr = cv2.imread(str(self.img_paths[idx]))

        # Crop
        if self.is_train:
            patch = self._random_crop(img_bgr, self.patch_size)
            # Random horizontal flip
            if random.random() > 0.5:
                patch = cv2.flip(patch, 1)
        else:
            patch = self._center_crop(img_bgr, self.patch_size)

        # Clean target
        clean = patch.copy()

        # Corrupted input
        corrupted = self._apply_random_corruption(patch)

        # BGR -> RGB, HWC -> CHW, normalize to [0,1]
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corrupted = cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        clean = torch.from_numpy(clean).permute(2, 0, 1)
        corrupted = torch.from_numpy(corrupted).permute(2, 0, 1)

        return corrupted, clean


# ──────────────────────────────────────────────
# SSIM Loss
# ──────────────────────────────────────────────
def _gaussian_kernel(size: int = 11, sigma: float = 1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = pred.size(1)

    kernel = _gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.expand(channels, 1, -1, -1)

    mu1 = nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu2 = nn.functional.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = nn.functional.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = nn.functional.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class CombinedLoss(nn.Module):
    """L1 + (1 - SSIM) weighted loss."""

    def __init__(self, ssim_weight: float = 0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - ssim(pred, target)
        return l1_loss + self.ssim_weight * ssim_loss


# ──────────────────────────────────────────────
# Validation metrics
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = nn.functional.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_psnr = 0.0
    total_ssim_val = 0.0
    n = 0

    for corrupted, clean in val_loader:
        corrupted = corrupted.to(DEVICE)
        clean = clean.to(DEVICE)

        restored = model(corrupted)

        total_psnr += compute_psnr(restored, clean) * corrupted.size(0)
        total_ssim_val += float(ssim(restored, clean)) * corrupted.size(0)
        n += corrupted.size(0)

    return total_psnr / n, total_ssim_val / n


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def save_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Training image restoration model (U-Net)", flush=True)
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Patch: {PATCH_SIZE}x{PATCH_SIZE}", flush=True)
    print(f"Corruptions: Noise(s={NOISE_SIGMA}), Blur(k={BLUR_KERNEL}), LowRes(f={DOWNSCALE_FACTOR})\n", flush=True)

    # Datasets
    train_ds = RestorationDataset(TRAIN_IMG_DIR, PATCH_SIZE, is_train=True)
    val_ds = RestorationDataset(VAL_IMG_DIR, PATCH_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images", flush=True)

    # Model
    model = RestorationUNet(channels=(32, 64, 128, 256))
    model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M\n", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = CombinedLoss(ssim_weight=0.3)

    # Logging
    history_path = OUT_DIR / "history.jsonl"
    best_ckpt = OUT_DIR / "best.pth"
    last_ckpt = OUT_DIR / "last.pth"
    best_psnr = 0.0

    start_time = time.time()
    n_batches = len(train_loader)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for i, (corrupted, clean) in enumerate(train_loader):
            corrupted = corrupted.to(DEVICE)
            clean = clean.to(DEVICE)

            restored = model(corrupted)
            loss = criterion(restored, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

            if (i + 1) % 200 == 0 or (i + 1) == n_batches:
                print(f"  [Epoch {epoch:03d}] batch {i+1}/{n_batches}  loss={loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Validate every 5 epochs
        val_psnr, val_ssim_val = 0.0, 0.0
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_psnr, val_ssim_val = validate(model, val_loader)
            print(f"[Epoch {epoch:03d}/{EPOCHS}] loss={avg_loss:.4f}  "
                  f"val_PSNR={val_psnr:.2f}dB  val_SSIM={val_ssim_val:.4f}", flush=True)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({"model": model.state_dict(), "epoch": epoch,
                             "psnr": val_psnr, "ssim": val_ssim_val}, best_ckpt)
                print(f"  -> New best PSNR: {val_psnr:.2f}dB", flush=True)
        else:
            print(f"[Epoch {epoch:03d}/{EPOCHS}] loss={avg_loss:.4f}", flush=True)

        log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_psnr": val_psnr if val_psnr > 0 else None,
            "val_ssim": val_ssim_val if val_ssim_val > 0 else None,
            "elapsed_sec": int(time.time() - start_time),
        }
        save_jsonl(history_path, log)
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

    elapsed = time.time() - start_time
    print(f"\nTraining done. Total time: {elapsed/60:.1f} min", flush=True)
    print(f"Best PSNR: {best_psnr:.2f}dB", flush=True)
    print(f"Best checkpoint: {best_ckpt.resolve()}", flush=True)


if __name__ == "__main__":
    main()
