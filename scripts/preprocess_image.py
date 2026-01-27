import csv
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

PROCESSED_DIR = Path("data/processed")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def process_split(split_name, transform):
    csv_path = PROCESSED_DIR / f"{split_name}_split.csv"
    out_dir = PROCESSED_DIR / split_name / "images_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            guid = row["guid"]

            # ★ 关键修复：字符串 → Path
            img_path = Path(row["image_path"])

            if not img_path.exists():
                print(f"[WARN] Image not found: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img)

            torch.save(tensor, out_dir / f"{guid}.pt")
            count += 1

    print(f"{split_name} image preprocessing done. Saved {count} files.")

def main():
    process_split("train", train_transform)
    process_split("val", val_transform)

if __name__ == "__main__":
    main()
