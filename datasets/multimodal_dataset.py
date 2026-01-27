import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from transformers import BertTokenizer

from config.label_map import LABEL2ID


class MultimodalDataset(Dataset):
    #返回同一个 guid 对应的：文本（tokenized）\图像张量\标签

    def __init__(
        self,
        split_csv: str,
        processed_root: str,
        bert_model_name: str = "bert-base-uncased",
        max_length: int = 128
    ):
        self.data = pd.read_csv(split_csv)
        self.processed_root = Path(processed_root)

        self.text_dir = self.processed_root / "texts_cleaned"
        self.image_dir = self.processed_root / "images_processed"

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        guid = row["guid"]
        label = LABEL2ID[row["label"]]

        # ===== Text =====
        text_path = self.text_dir / f"{guid}.txt"
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # squeeze: [1, L] -> [L]
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ===== Image =====
        image_path = self.image_dir / f"{guid}.pt"
        image = torch.load(image_path)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }
