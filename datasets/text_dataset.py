import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.label_map import LABEL2ID

class TextDataset(Dataset):
    def __init__(self, split_csv, processed_root, max_length=128):
        
        #split_csv: data/processed/train_split.csv 或 val_split.csv
        #processed_root: data/processed/train 或 data/processed/val
        
        self.samples = []
        self.processed_root = Path(processed_root)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

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

        text_path = self.processed_root / "texts_cleaned" / f"{guid}.txt"

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
