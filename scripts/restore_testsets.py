"""
Apply trained restoration model to corrupted test sets.

Generates restored versions of Test_Noise, Test_Blur, Test_LowRes
in both COCO and YOLO formats for evaluation.

Usage:
    python -m scripts.restore_testsets
"""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.restoration_net import RestorationUNet

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CKPT_PATH = Path("experiments/restoration/best.pth")

COCO_TESTSET_ROOT = Path("data/testsets/coco6")
YOLO_TESTSET_ROOT = Path("data/testsets/yolo6")
COCO_OUT_ROOT = Path("data/testsets/coco6_restored")
YOLO_OUT_ROOT = Path("data/testsets/yolo6_restored")

VARIANTS_TO_RESTORE = ["Test_Noise", "Test_Blur", "Test_LowRes"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(ckpt_path: Path):
    model = RestorationUNet(channels=(32, 64, 128, 256))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    psnr = ckpt.get("psnr", "N/A")
    ssim_val = ckpt.get("ssim", "N/A")
    print(f"Loaded model: PSNR={psnr}, SSIM={ssim_val}")
    return model


@torch.no_grad()
def restore_image(model, img_bgr: np.ndarray) -> np.ndarray:
    """Restore a single BGR image using the model."""
    h, w = img_bgr.shape[:2]

    # Pad to multiple of 16 for U-Net compatibility
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # BGR -> RGB, normalize, to tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    restored = model(tensor)

    # Back to numpy BGR
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored = (restored * 255.0).clip(0, 255).astype(np.uint8)
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        restored = restored[:h, :w]

    return restored


def restore_variant_coco(model, variant: str):
    """Restore COCO format test set images."""
    src_img_dir = COCO_TESTSET_ROOT / variant / "images" / "val"
    src_ann_dir = COCO_TESTSET_ROOT / variant / "annotations"
    dst_img_dir = COCO_OUT_ROOT / variant / "images" / "val"
    dst_ann_dir = COCO_OUT_ROOT / variant / "annotations"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    # Copy annotations (unchanged)
    for ann_file in src_ann_dir.glob("*.json"):
        shutil.copy2(ann_file, dst_ann_dir / ann_file.name)

    # Restore images
    img_files = sorted(src_img_dir.glob("*.jpg"))
    for i, img_path in enumerate(img_files):
        img_bgr = cv2.imread(str(img_path))
        restored = restore_image(model, img_bgr)
        cv2.imwrite(str(dst_img_dir / img_path.name), restored)

        if (i + 1) % 100 == 0 or (i + 1) == len(img_files):
            print(f"    COCO {variant}: {i+1}/{len(img_files)}", flush=True)


def restore_variant_yolo(model, variant: str):
    """Restore YOLO format test set images."""
    src_img_dir = YOLO_TESTSET_ROOT / variant / "images" / "val"
    src_label_dir = YOLO_TESTSET_ROOT / variant / "labels" / "val"
    dst_img_dir = YOLO_OUT_ROOT / variant / "images" / "val"
    dst_label_dir = YOLO_OUT_ROOT / variant / "labels" / "val"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    # Copy labels (unchanged)
    if src_label_dir.exists():
        for lbl_file in src_label_dir.glob("*.txt"):
            shutil.copy2(lbl_file, dst_label_dir / lbl_file.name)

    # Copy data.yaml and update paths
    src_yaml = YOLO_TESTSET_ROOT / variant / "data.yaml"
    dst_yaml = YOLO_OUT_ROOT / variant / "data.yaml"
    if src_yaml.exists():
        content = src_yaml.read_text(encoding="utf-8")
        # Update path to point to restored directory
        content = content.replace(str(YOLO_TESTSET_ROOT), str(YOLO_OUT_ROOT))
        content = content.replace(YOLO_TESTSET_ROOT.as_posix(), YOLO_OUT_ROOT.as_posix())
        # If relative paths, just copy as-is and we'll handle in eval
        dst_yaml.parent.mkdir(parents=True, exist_ok=True)
        dst_yaml.write_text(content, encoding="utf-8")

    # Restore images
    img_files = sorted(src_img_dir.glob("*.jpg"))
    for i, img_path in enumerate(img_files):
        img_bgr = cv2.imread(str(img_path))
        restored = restore_image(model, img_bgr)
        cv2.imwrite(str(dst_img_dir / img_path.name), restored)

        if (i + 1) % 100 == 0 or (i + 1) == len(img_files):
            print(f"    YOLO {variant}: {i+1}/{len(img_files)}", flush=True)


def copy_clean_testsets():
    """Copy Test_Clean as-is (no restoration needed)."""
    # COCO
    src = COCO_TESTSET_ROOT / "Test_Clean"
    dst = COCO_OUT_ROOT / "Test_Clean"
    if not dst.exists():
        shutil.copytree(str(src), str(dst))
        print("  Copied Test_Clean (COCO)")

    # YOLO
    src = YOLO_TESTSET_ROOT / "Test_Clean"
    dst = YOLO_OUT_ROOT / "Test_Clean"
    if not dst.exists():
        shutil.copytree(str(src), str(dst))
        print("  Copied Test_Clean (YOLO)")


def main():
    print(f"Device: {DEVICE}")
    print(f"Restoring corrupted test sets ...\n")

    model = build_model(CKPT_PATH)

    # Copy clean test sets
    print("Copying Test_Clean ...", flush=True)
    copy_clean_testsets()

    # Restore corrupted test sets
    for variant in VARIANTS_TO_RESTORE:
        print(f"\n  Restoring {variant} ...", flush=True)
        restore_variant_coco(model, variant)
        restore_variant_yolo(model, variant)

    print(f"\nAll restored test sets saved.")
    print(f"  COCO: {COCO_OUT_ROOT.resolve()}")
    print(f"  YOLO: {YOLO_OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
