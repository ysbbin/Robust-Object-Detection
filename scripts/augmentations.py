"""
Shared corruption augmentations for robustness training.

Parameters match build_corrupted_testsets.py exactly so that
training augmentations are consistent with test-set corruptions.
"""

import random
import numpy as np
import cv2
from PIL import Image

# ── Parameters (same as build_corrupted_testsets.py) ─────────
NOISE_SIGMA = 15
BLUR_KERNEL = 9
BLUR_ANGLE_DEG = 0
DOWNSCALE_FACTOR = 0.5


# ── Low-level corruption functions ───────────────────────────
def _motion_blur_kernel(k: int, angle_deg: float):
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def apply_noise(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(img_bgr: np.ndarray, k: int, angle_deg: float) -> np.ndarray:
    kernel = _motion_blur_kernel(k, angle_deg)
    return cv2.filter2D(img_bgr, -1, kernel)


def apply_lowres(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_random_corruption(img_bgr: np.ndarray) -> np.ndarray:
    """Apply one random corruption to a BGR numpy image."""
    choice = random.choice(["noise", "blur", "lowres"])
    if choice == "noise":
        return apply_noise(img_bgr, NOISE_SIGMA)
    elif choice == "blur":
        return apply_motion_blur(img_bgr, BLUR_KERNEL, BLUR_ANGLE_DEG)
    else:
        return apply_lowres(img_bgr, DOWNSCALE_FACTOR)


# ── FRCNN: PIL Image transform ──────────────────────────────
class RandomCorruption:
    """
    PIL Image transform that randomly applies one corruption.
    Use in torchvision transforms pipeline for Faster R-CNN.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        arr = _apply_random_corruption(arr)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ── Ultralytics: monkey-patch Albumentations ─────────────────
def patch_ultralytics_augmentations():
    """
    Inject corruption augmentations into Ultralytics' Albumentations pipeline.
    Call this ONCE before model.train().

    Works by monkey-patching Albumentations.__call__ to apply a random
    corruption (noise/blur/lowres) with 50% probability on labels["img"]
    before the original albumentations transforms run.
    """
    from ultralytics.data import augment as _augment

    _OrigCall = _augment.Albumentations.__call__

    def _patched_call(self, labels):
        # Apply corruption with 50% probability (pixel-only, no bbox change)
        if random.random() < 0.5:
            labels["img"] = _apply_random_corruption(labels["img"])
        return _OrigCall(self, labels)

    _augment.Albumentations.__call__ = _patched_call
    print("[augmentations] Ultralytics Albumentations patched with corruption augmentations")
