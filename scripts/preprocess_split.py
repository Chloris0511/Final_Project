# 数据划分

import random
import csv
from pathlib import Path

RANDOM_SEED = 42
TRAIN_RATIO = 0.8

RAW_ROOT = Path("data/raw")
SAMPLE_DIR = RAW_ROOT / "data"   # ★
TRAIN_LABEL_FILE = RAW_ROOT / "train.txt"

PROCESSED_DIR = Path("data/processed")
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

def find_image_file(guid):
    for ext in IMAGE_EXTS:
        img_path = SAMPLE_DIR / f"{guid}{ext}"
        if img_path.exists():
            return img_path
    return None

def main():
    random.seed(RANDOM_SEED)

    samples = []

    with open(TRAIN_LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            guid, label = line.strip().split(",")

            text_path = SAMPLE_DIR / f"{guid}.txt"
            image_path = find_image_file(guid)

            if not text_path.exists():
                continue
            if image_path is None:
                continue

            samples.append({
                "guid": guid,
                "label": label,
                "text_path": str(text_path),
                "image_path": str(image_path)
            })

    print(f"[INFO] Valid samples found: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError(
            "No valid samples found. Please check SAMPLE_DIR path!"
        )

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_RATIO)

    train_set = samples[:split_idx]
    val_set = samples[split_idx:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["guid", "label", "text_path", "image_path"]
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv(PROCESSED_DIR / "train_split.csv", train_set)
    write_csv(PROCESSED_DIR / "val_split.csv", val_set)

    print(f"[INFO] Train split: {len(train_set)}")
    print(f"[INFO] Val split: {len(val_set)}")

if __name__ == "__main__":
    main()

# 文本和图像要按 split 物理分开！