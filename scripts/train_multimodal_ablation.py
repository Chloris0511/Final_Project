import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel
from torchvision.models import resnet18
from tqdm import tqdm

from datasets.multimodal_attention_dataset import MultimodalAttentionDataset
from models.multimodal_attention_model import AttentionFusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 超参数 =====
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-5
HIDDEN_DIM = 256
NUM_CLASSES = 3


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def evaluate(model, loader, ablation):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["image"].to(DEVICE),
                ablation=ablation
            )
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch["label"].tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def main():
    train_dataset = MultimodalAttentionDataset(
        "data/processed/train_split.csv",
        "data/processed"
    )
    val_dataset = MultimodalAttentionDataset(
        "data/processed/val_split.csv",
        "data/processed"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    image_encoder = ResNetEncoder()

    result_dir = Path("outputs/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "val_metrics.csv"

    for ablation in [None, "text", "image"]:
        ablation_name = ablation or "multimodal"
        print(f"\n===== 开始训练 Ablation: {ablation_name} =====")

        model = AttentionFusionModel(
            text_encoder,
            image_encoder,
            768,
            image_encoder.feature_dim,
            HIDDEN_DIM,
            NUM_CLASSES
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # ===== 训练 =====
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            preds, labels = [], []

            loop = tqdm(
                train_loader,
                desc=f"Ablation {ablation_name} | Epoch {epoch+1}/{EPOCHS}"
            )

            for batch in loop:
                optimizer.zero_grad()

                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["image"].to(DEVICE),
                    ablation=ablation
                )

                loss = criterion(
                    logits,
                    batch["label"].to(DEVICE)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.detach().cpu().tolist())
                labels.extend(batch["label"].tolist())

                loop.set_postfix(loss=loss.item())

            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(labels, preds)

            print(
                f"[Epoch {epoch+1}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}"
            )

        # ===== 验证 =====
        acc, f1 = evaluate(model, val_loader, ablation)
        print(
            f"【Val Result】 Ablation={ablation_name} | "
            f"Acc={acc:.4f}, Macro-F1={f1:.4f}"
        )

        # ===== 写入 CSV =====
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["model", "ablation", "val_acc", "val_macro_f1"])
            writer.writerow([
                "attention_fusion",
                ablation_name,
                f"{acc:.4f}",
                f"{f1:.4f}"
            ])


if __name__ == "__main__":
    main()
