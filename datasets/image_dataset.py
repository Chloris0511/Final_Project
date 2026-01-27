import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from config.label_map import LABEL2ID

class ImageDataset(Dataset):
    def __init__(self, split_csv, processed_root):
        """
        split_csv: data/processed/train_split.csv »ò val_split.csv
        processed_root: data/processed/train »ò data/processed/val
        """
        self.samples = []
        self.processed_root = Path(processed_root)

        with open(split_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "guid": row["guid"],
                    "label": LABEL2ID[row["label"]]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        guid = sample["guid"]
        label = sample["label"]

        image_path = self.processed_root / "images_processed" / f"{guid}.pt"
        image_tensor = torch.load(image_path)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }
