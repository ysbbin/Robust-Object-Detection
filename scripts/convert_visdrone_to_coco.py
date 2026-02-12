import json
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image

from scripts.paths import TRAIN_DIR, VAL_DIR, check_dataset

# === VisDrone classes to use ===
USED_CLASSES = [1, 4, 5, 6, 9, 10]

# COCO category_id는 보통 1..K로 두는 게 안전합니다 (mmdet/torchvision 호환)
# (VisDrone 원본 id) -> (COCO category_id: 1..6)
CLASS_MAP_COCO = {
    1: 1,   # pedestrian
    4: 2,   # car
    5: 3,   # van
    6: 4,   # truck
    9: 5,   # bus
    10: 6,  # motor
}

# category_id(1..6) 순서에 맞춘 클래스 이름
CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "van"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
    {"id": 6, "name": "motor"},
]

OUT_ROOT = Path("data/processed/visdrone_coco6")


def ensure_dirs():
    (OUT_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotations").mkdir(parents=True, exist_ok=True)


def parse_line(line: str):
    """
    VisDrone annotation format (typical):
    x, y, w, h, score, class_id, truncation, occlusion
    """
    parts = line.strip().split(",")
    if len(parts) < 8:
        return None
    x, y, w, h = map(float, parts[0:4])
    score = float(parts[4]) if parts[4] != "" else 1.0
    cls = int(parts[5])
    return x, y, w, h, cls, score


def find_image_path(img_dir: Path, stem: str):
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def clamp_box_xywh(x, y, w, h, W, H):
    # xywh -> x1y1x2y2
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # clamp
    x1 = max(0.0, min(x1, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))

    nw = x2 - x1
    nh = y2 - y1
    return x1, y1, nw, nh


def build_coco_for_split(split_name: str, split_dir: Path):
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    images = []
    annotations = []
    box_counter = Counter()

    image_id = 1
    ann_id = 1

    kept_images = 0
    empty_images = 0
    removed_invalid = 0
    removed_filtered = 0

    # VisDrone annotations are per-image .txt
    for ann_path in ann_dir.glob("*.txt"):
        img_path = find_image_path(img_dir, ann_path.stem)
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        # COCO image entry
        images.append({
            "id": image_id,
            "file_name": img_path.name,  # we will copy image with same filename
            "width": W,
            "height": H,
        })

        # copy image
        out_img_path = OUT_ROOT / "images" / split_name / img_path.name
        shutil.copy2(img_path, out_img_path)

        # read and convert anns
        has_any = False

        for line in ann_path.read_text().splitlines():
            parsed = parse_line(line)
            if parsed is None:
                continue

            x, y, w, h, cls, score = parsed

            # (선택) score==0 박스 제거하고 싶으면 아래 주석 해제
            if score <= 0:
                removed_filtered += 1
                continue

            if cls not in USED_CLASSES:
                removed_filtered += 1
                continue

            if w <= 0 or h <= 0:
                removed_invalid += 1
                continue

            x, y, w, h = clamp_box_xywh(x, y, w, h, W, H)
            if w <= 0 or h <= 0:
                removed_invalid += 1
                continue

            coco_cat = CLASS_MAP_COCO[cls]
            area = float(w * h)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": coco_cat,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": area,
                "iscrowd": 0,
            })

            ann_id += 1
            has_any = True
            box_counter[cls] += 1

        if has_any:
            kept_images += 1
        else:
            empty_images += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_json = OUT_ROOT / "annotations" / f"instances_{split_name}.json"
    out_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_json": out_json,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "box_count_by_original_cls": box_counter,
        "kept_images": kept_images,
        "empty_images": empty_images,
        "removed_invalid": removed_invalid,
        "removed_filtered": removed_filtered,
    }


def main():
    check_dataset()
    ensure_dirs()

    print("Converting COCO train...")
    train_info = build_coco_for_split("train", TRAIN_DIR)

    print("Converting COCO val...")
    val_info = build_coco_for_split("val", VAL_DIR)

    print("\n=== COCO Summary (train) ===")
    print("json:", train_info["out_json"])
    print("images:", train_info["num_images"], "annotations:", train_info["num_annotations"])
    print("box count (original class ids):", train_info["box_count_by_original_cls"])
    print("kept images:", train_info["kept_images"],
          "| empty images:", train_info["empty_images"],
          "| invalid boxes removed:", train_info["removed_invalid"],
          "| filtered boxes removed:", train_info["removed_filtered"])

    print("\n=== COCO Summary (val) ===")
    print("json:", val_info["out_json"])
    print("images:", val_info["num_images"], "annotations:", val_info["num_annotations"])
    print("box count (original class ids):", val_info["box_count_by_original_cls"])
    print("kept images:", val_info["kept_images"],
          "| empty images:", val_info["empty_images"],
          "| invalid boxes removed:", val_info["removed_invalid"],
          "| filtered boxes removed:", val_info["removed_filtered"])

    print("\n✅ COCO dataset created at:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
