import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from transformers import BertTokenizer

from config.label_map import LABEL2ID

class MultimodalAttentionDataset(Dataset):
    def __init__(self, split_csv: str, processed_root: str, bert_model_name="bert-base-uncased", max_length=128):
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

        # 文本
        text_path = self.text_dir / f"{guid}.txt"
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        encoding = self.tokenizer(text, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 图像
        image_path = self.image_dir / f"{guid}.pt"
        image = torch.load(image_path)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }
