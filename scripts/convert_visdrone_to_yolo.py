from pathlib import Path
from PIL import Image
import shutil
from collections import Counter

from scripts.paths import TRAIN_DIR, VAL_DIR, check_dataset

# === VisDrone classes to use ===
USED_CLASSES = [1, 4, 5, 6, 9, 10]

CLASS_MAP = {
    1: 0,   # pedestrian
    4: 1,   # car
    5: 2,   # van
    6: 3,   # truck
    9: 4,   # bus
    10: 5,  # motor
}

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]

# Output root (keep folder, overwrite contents)
OUT_ROOT = Path("data/processed/visdrone_yolo6")

# 정책: 사용 클래스가 없어도 이미지는 포함(빈 라벨 생성)
KEEP_EMPTY_IMAGES = True

def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def parse_line(line: str):
    parts = line.strip().split(",")
    if len(parts) < 8:
        return None
    x, y, w, h = map(float, parts[0:4])
    cls = int(parts[5])
    # parts[4] = score (often 1 in train/val), parts[6]=truncation, parts[7]=occlusion
    score = float(parts[4]) if parts[4] != "" else 1.0
    return x, y, w, h, cls, score

def clamp_box(x, y, w, h, W, H):
    # convert to x1,y1,x2,y2
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # clamp to image boundaries
    x1 = max(0.0, min(x1, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))

    nw = x2 - x1
    nh = y2 - y1
    return x1, y1, nw, nh

def convert_split(split_name, split_dir):
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    counter = Counter()
    kept_images = 0
    empty_images = 0
    removed_invalid = 0

    for ann_path in ann_dir.glob("*.txt"):
        # image extension can vary; try common ones
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            p = img_dir / (ann_path.stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        yolo_lines = []

        for line in ann_path.read_text().splitlines():
            parsed = parse_line(line)
            if parsed is None:
                continue

            x, y, w, h, cls, score = parsed

            # (선택) score==0인 박스 제외하고 싶으면 아래 주석 해제
            if score <= 0:
                continue

            if cls not in USED_CLASSES:
                continue

            if w <= 0 or h <= 0:
                removed_invalid += 1
                continue

            # clamp
            x, y, w, h = clamp_box(x, y, w, h, W, H)
            if w <= 0 or h <= 0:
                removed_invalid += 1
                continue

            x_c = (x + w / 2) / W
            y_c = (y + h / 2) / H
            w_n = w / W
            h_n = h / H

            # final clamp to [0,1]
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            w_n = min(max(w_n, 0.0), 1.0)
            h_n = min(max(h_n, 0.0), 1.0)

            yolo_cls = CLASS_MAP[cls]
            yolo_lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            counter[cls] += 1

        out_img = OUT_ROOT / "images" / split_name / img_path.name
        out_lbl = OUT_ROOT / "labels" / split_name / (ann_path.stem + ".txt")

        # copy image always if KEEP_EMPTY_IMAGES
        if yolo_lines:
            shutil.copy2(img_path, out_img)
            out_lbl.write_text("\n".join(yolo_lines))
            kept_images += 1
        else:
            if KEEP_EMPTY_IMAGES:
                shutil.copy2(img_path, out_img)
                out_lbl.write_text("")  # empty label
                empty_images += 1

    return counter, kept_images, empty_images, removed_invalid

def write_data_yaml():
    yaml_path = OUT_ROOT / "data.yaml"
    lines = [
        f"path: {OUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:"
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines))

def main():
    check_dataset()
    ensure_dirs()

    print("Converting train...")
    train_stats, train_kept, train_empty, train_removed = convert_split("train", TRAIN_DIR)

    print("Converting val...")
    val_stats, val_kept, val_empty, val_removed = convert_split("val", VAL_DIR)

    write_data_yaml()

    print("\n=== Box count (train, original class ids) ===")
    print(train_stats)
    print("kept images:", train_kept, "| empty images kept:", train_empty, "| invalid boxes removed:", train_removed)

    print("\n=== Box count (val, original class ids) ===")
    print(val_stats)
    print("kept images:", val_kept, "| empty images kept:", val_empty, "| invalid boxes removed:", val_removed)

    print("\n✅ YOLO dataset created at:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
