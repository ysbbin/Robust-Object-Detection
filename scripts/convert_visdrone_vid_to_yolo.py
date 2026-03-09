"""
Convert VisDrone-VID dataset to YOLO format.

VisDrone-VID annotation format (per sequence .txt):
  frame_index, track_id, x, y, w, h, score, category, truncation, occlusion

Output YOLO format:
  data/processed/visdrone_vid_yolo6/
  ├── images/
  │   ├── train/   {seq}_{frame_id:07d}.jpg
  │   └── val/
  ├── labels/
  │   ├── train/   {seq}_{frame_id:07d}.txt
  │   └── val/
  └── data.yaml

Usage:
    python -m scripts.convert_visdrone_vid_to_yolo
"""

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_ROOT = Path(r"C:/Users/PC/Desktop/AI Project/Dataset")
VID_TRAIN = DATA_ROOT / "VisDrone2019-VID-train"
VID_VAL   = DATA_ROOT / "VisDrone2019-VID-val"

OUT_ROOT  = Path("data/processed/visdrone_vid_yolo6")

# ──────────────────────────────────────────────
# Class mapping (same as DET project)
# ──────────────────────────────────────────────
USED_CLASSES = {1, 4, 5, 6, 9, 10}

CLASS_MAP = {
    1: 0,   # pedestrian
    4: 1,   # car
    5: 2,   # van
    6: 3,   # truck
    9: 4,   # bus
    10: 5,  # motor
}

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]

# 사용 클래스 없는 프레임도 포함 (빈 라벨)
KEEP_EMPTY_FRAMES = True


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def clamp_box(x, y, w, h, W, H):
    x1 = max(0.0, min(float(x), float(W)))
    y1 = max(0.0, min(float(y), float(H)))
    x2 = max(0.0, min(float(x + w), float(W)))
    y2 = max(0.0, min(float(y + h), float(H)))
    return x1, y1, x2 - x1, y2 - y1


def parse_annotation(ann_path: Path) -> defaultdict:
    """
    Returns dict: frame_index -> list of (x, y, w, h, category)
    Filters out score==0, unused classes, invalid boxes.
    """
    frame_dict = defaultdict(list)
    removed = 0

    for line in ann_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue

        frame_idx = int(parts[0])
        # track_id = parts[1]  (not used for VOD)
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        score      = float(parts[6]) if parts[6].strip() != "" else 1.0
        category   = int(parts[7])

        if score <= 0:
            continue
        if category not in USED_CLASSES:
            continue
        if w <= 0 or h <= 0:
            removed += 1
            continue

        frame_dict[frame_idx].append((x, y, w, h, category))

    return frame_dict, removed


# ──────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────
def convert_split(split_name: str, split_dir: Path):
    seq_dir = split_dir / "sequences"
    ann_dir = split_dir / "annotations"

    if not seq_dir.exists():
        raise FileNotFoundError(f"sequences dir not found: {seq_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"annotations dir not found: {ann_dir}")

    counter         = Counter()
    kept_frames     = 0
    empty_frames    = 0
    total_removed   = 0
    skipped_frames  = 0

    sequences = sorted(ann_dir.glob("*.txt"))
    print(f"  [{split_name}] {len(sequences)} sequences found")

    for ann_path in sequences:
        seq_name   = ann_path.stem                    # e.g. uav0000086_00000_v
        frames_dir = seq_dir / seq_name

        if not frames_dir.exists():
            print(f"  [WARN] sequence folder missing: {frames_dir}")
            continue

        frame_dict, removed = parse_annotation(ann_path)
        total_removed += removed

        # iterate over all jpg frames in the sequence folder
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.png"))

        for frame_path in frame_files:
            # frame_id from filename: "0000001.jpg" → 1
            frame_id = int(frame_path.stem)

            # output filename: {seq}_{frame_id:07d}
            out_stem = f"{seq_name}_{frame_id:07d}"
            out_img  = OUT_ROOT / "images" / split_name / (out_stem + ".jpg")
            out_lbl  = OUT_ROOT / "labels" / split_name / (out_stem + ".txt")

            boxes = frame_dict.get(frame_id, [])

            # get image size (needed for normalization)
            try:
                with Image.open(frame_path) as im:
                    W, H = im.size
            except Exception:
                skipped_frames += 1
                continue

            yolo_lines = []
            for (x, y, w, h, cat) in boxes:
                x, y, w, h = clamp_box(x, y, w, h, W, H)
                if w <= 0 or h <= 0:
                    total_removed += 1
                    continue

                x_c = min(max((x + w / 2) / W, 0.0), 1.0)
                y_c = min(max((y + h / 2) / H, 0.0), 1.0)
                w_n = min(max(w / W, 0.0), 1.0)
                h_n = min(max(h / H, 0.0), 1.0)

                yolo_cls = CLASS_MAP[cat]
                yolo_lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                counter[cat] += 1

            if yolo_lines:
                shutil.copy2(frame_path, out_img)
                out_lbl.write_text("\n".join(yolo_lines))
                kept_frames += 1
            elif KEEP_EMPTY_FRAMES:
                shutil.copy2(frame_path, out_img)
                out_lbl.write_text("")
                empty_frames += 1

    return counter, kept_frames, empty_frames, total_removed, skipped_frames


# ──────────────────────────────────────────────
# data.yaml
# ──────────────────────────────────────────────
def write_data_yaml():
    yaml_path = OUT_ROOT / "data.yaml"
    lines = [
        f"path: {OUT_ROOT.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines))
    print(f"  data.yaml → {yaml_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def check_dataset():
    for name, path in [("VID-train", VID_TRAIN), ("VID-val", VID_VAL)]:
        if not (path / "sequences").exists() or not (path / "annotations").exists():
            raise FileNotFoundError(
                f"[{name}] dataset not found at: {path}\n"
                f"Expected: {path}/sequences  and  {path}/annotations"
            )
    print("✅ VisDrone-VID dataset structure is valid.\n")


def main():
    check_dataset()
    ensure_dirs()

    print("Converting train ...")
    tr_cnt, tr_kept, tr_empty, tr_rm, tr_skip = convert_split("train", VID_TRAIN)

    print("\nConverting val ...")
    vl_cnt, vl_kept, vl_empty, vl_rm, vl_skip = convert_split("val", VID_VAL)

    write_data_yaml()

    name_map = {1: "pedestrian", 4: "car", 5: "van", 6: "truck", 9: "bus", 10: "motor"}

    print("\n=== Train ===")
    print(f"  kept frames : {tr_kept}")
    print(f"  empty frames: {tr_empty}")
    print(f"  invalid boxes removed: {tr_rm}")
    print(f"  skipped frames (read error): {tr_skip}")
    print("  box counts:", {name_map[k]: v for k, v in sorted(tr_cnt.items())})

    print("\n=== Val ===")
    print(f"  kept frames : {vl_kept}")
    print(f"  empty frames: {vl_empty}")
    print(f"  invalid boxes removed: {vl_rm}")
    print(f"  skipped frames (read error): {vl_skip}")
    print("  box counts:", {name_map[k]: v for k, v in sorted(vl_cnt.items())})

    print(f"\n✅ YOLO dataset created at: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
