import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from datasets.multimodal_attention_dataset import MultimodalAttentionDataset
from models.multimodal_attention_model import AttentionFusionModel
from utils.metrics import compute_metrics
from config.baseline_config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from config.label_map import LABEL2ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss = 0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for batch in bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds

def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    train_set = MultimodalAttentionDataset(
        split_csv="data/processed/train_split.csv",
        processed_root="data/processed/train"
    )
    val_set = MultimodalAttentionDataset(
        split_csv="data/processed/val_split.csv",
        processed_root="data/processed/val"
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionFusionModel(num_labels=len(LABEL2ID)).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("===== 开始训练 Attention Fusion 模型 =====")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch + 1, EPOCHS)
        y_true, y_pred = validate(model, val_loader)
        metrics = compute_metrics(y_true, y_pred)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f}"
        )

    torch.save(model.state_dict(), "outputs/checkpoints/attention_fusion.pt")
    with open("outputs/metrics/attention_fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("训练完成，已保存模型 & 指标。")

if __name__ == "__main__":
    main()
