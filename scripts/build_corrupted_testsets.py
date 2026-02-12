import os
from pathlib import Path
import shutil
import numpy as np
import cv2

# ====== 경로 설정 ======
YOLO_SRC = Path("data/processed/visdrone_yolo6")      # 원본 YOLO6
COCO_SRC = Path("data/processed/visdrone_coco6")      # 원본 COCO6
OUT_ROOT = Path("data/testsets")                      # 출력 루트

# ====== 열화 파라미터(필요하면 여기만 바꾸면 됨) ======
SEED = 42

# Gaussian noise (std)
NOISE_SIGMA = 15  # 5/10/20 등으로 단계 확장 가능

# Motion blur
BLUR_KERNEL = 9   # 5/9/13 등으로 단계 확장 가능
BLUR_ANGLE_DEG = 0  # 0~180. (0이면 가로 방향)

# Low resolution
DOWNSCALE_FACTOR = 0.5  # 0.5면 절반으로 줄였다가 다시 키움


def set_seed(seed: int):
    np.random.seed(seed)


def motion_blur_kernel(k: int, angle_deg: float):
    """Create a motion blur kernel of size k at given angle."""
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    # rotate
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def apply_noise(img_bgr: np.ndarray, sigma: float):
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_motion_blur(img_bgr: np.ndarray, k: int, angle_deg: float):
    kernel = motion_blur_kernel(k, angle_deg)
    out = cv2.filter2D(img_bgr, -1, kernel)
    return out


def apply_lowres(img_bgr: np.ndarray, factor: float):
    h, w = img_bgr.shape[:2]
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_yolo_valonly_yaml(dst_root: Path):
    """Create a data.yaml that evaluates on this testset val folder."""
    yaml_path = dst_root / "data.yaml"
    lines = [
        f"path: {dst_root.as_posix()}",
        "train: images/val",   # 학습용으로 쓰지 않을 거지만 형식상 채움
        "val: images/val",
        "",
        "names:",
        "  0: pedestrian",
        "  1: car",
        "  2: van",
        "  3: truck",
        "  4: bus",
        "  5: motor",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def build_yolo_testsets():
    src_img_dir = YOLO_SRC / "images" / "val"
    src_lbl_dir = YOLO_SRC / "labels" / "val"

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        raise FileNotFoundError("YOLO val images/labels not found. Check YOLO_SRC path.")

    variants = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]

    for v in variants:
        dst_root = OUT_ROOT / "yolo6" / v
        dst_img_dir = dst_root / "images" / "val"
        dst_lbl_dir = dst_root / "labels" / "val"
        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)

        # labels는 항상 동일 복사
        for lbl in src_lbl_dir.glob("*.txt"):
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)

        write_yolo_valonly_yaml(dst_root)

        # images 처리
        for img_path in src_img_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if v == "Test_Clean":
                out = img
            elif v == "Test_Noise":
                out = apply_noise(img, NOISE_SIGMA)
            elif v == "Test_Blur":
                out = apply_motion_blur(img, BLUR_KERNEL, BLUR_ANGLE_DEG)
            elif v == "Test_LowRes":
                out = apply_lowres(img, DOWNSCALE_FACTOR)
            else:
                out = img

            cv2.imwrite(str(dst_img_dir / img_path.name), out)

    print("✅ YOLO Test sets created:", (OUT_ROOT / "yolo6").resolve())


def build_coco_testsets():
    src_img_dir = COCO_SRC / "images" / "val"
    src_ann = COCO_SRC / "annotations" / "instances_val.json"

    if not src_img_dir.exists() or not src_ann.exists():
        raise FileNotFoundError("COCO val images or instances_val.json not found. Check COCO_SRC path.")

    variants = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]

    for v in variants:
        dst_root = OUT_ROOT / "coco6" / v
        dst_img_dir = dst_root / "images" / "val"
        dst_ann_dir = dst_root / "annotations"
        ensure_dir(dst_img_dir)
        ensure_dir(dst_ann_dir)

        # annotation json은 그대로 복사(파일명 동일, bbox 동일)
        shutil.copy2(src_ann, dst_ann_dir / "instances_val.json")

        for img_path in src_img_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if v == "Test_Clean":
                out = img
            elif v == "Test_Noise":
                out = apply_noise(img, NOISE_SIGMA)
            elif v == "Test_Blur":
                out = apply_motion_blur(img, BLUR_KERNEL, BLUR_ANGLE_DEG)
            elif v == "Test_LowRes":
                out = apply_lowres(img, DOWNSCALE_FACTOR)
            else:
                out = img

            cv2.imwrite(str(dst_img_dir / img_path.name), out)

    print("✅ COCO Test sets created:", (OUT_ROOT / "coco6").resolve())


def main():
    set_seed(SEED)
    build_yolo_testsets()
    build_coco_testsets()
    print("\n✅ All corrupted test sets are ready under:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()

