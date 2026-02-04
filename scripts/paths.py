from pathlib import Path

# === VisDrone dataset root (HOME PC) ===
DATA_ROOT = Path(
    r"C:/Users/ysb53/Desktop/AI Project/Dataset"
)

TRAIN_DIR = DATA_ROOT / "VisDrone2019-DET-train"
VAL_DIR   = DATA_ROOT / "VisDrone2019-DET-val"

def check_dataset():
    for split_name, split_dir in {
        "train": TRAIN_DIR,
        "val": VAL_DIR
    }.items():
        img_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        if not img_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(
                f"[{split_name}] dataset not found:\n"
                f"{img_dir}\n{ann_dir}"
            )
    print("✅ VisDrone dataset structure is valid.")
