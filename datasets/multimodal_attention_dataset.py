import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import BertTokenizer


class MultimodalAttentionDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        processed_root: str,
        bert_model_name="bert-base-uncased",
        max_length=128
    ):
        self.data = pd.read_csv(split_csv)
        self.processed_root = Path(processed_root)

        # ===== 自动判断 split =====
        split_name = Path(split_csv).name
        self.split = "train" if "train" in split_name else "val"

        self.text_dir = self.processed_root / self.split / "texts_cleaned"
        self.image_dir = self.processed_root / self.split / "images_processed"

        self.label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        guid = str(row["guid"])
        raw_label = row["label"]
        label = self.label_map[raw_label]

        # ===== 文本 =====
        text_path = self.text_dir / f"{guid}.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"缺少文本文件: {text_path}")

        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # ===== 图像 =====
        image_path = self.image_dir / f"{guid}.pt"
        if not image_path.exists():
            raise FileNotFoundError(f"缺少图像文件: {image_path}")

        with open(image_path, "rb") as f:
            image = torch.load(f)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": label
        }
