import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from datasets.image_dataset import ImageDataset
from models.image_model import ResNetImageClassifier
from utils.metrics import compute_metrics
from config.label_map import LABEL2ID
from config.baseline_config import BATCH_SIZE, LEARNING_RATE, EPOCHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False
    )

    for batch in progress_bar:
        optimizer.zero_grad()

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def main():
    
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    print("===== 开始训练 Image-only ResNet 模型 =====")

    # ===== Dataset =====
    train_dataset = ImageDataset(
        split_csv="data/processed/train_split.csv",
        processed_root="data/processed/train"
    )

    val_dataset = ImageDataset(
        split_csv="data/processed/val_split.csv",
        processed_root="data/processed/val"
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

    # ===== Model =====
    model = ResNetImageClassifier(
        num_labels=len(LABEL2ID)
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # ===== Training Loop =====
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch + 1,
            EPOCHS
        )

        y_true, y_pred = validate(model, val_loader)
        metrics = compute_metrics(y_true, y_pred)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f}"
        )

    # ===== Save =====
    torch.save(
        model.state_dict(),
        "outputs/checkpoints/image_only.pt"
    )

    with open(
        "outputs/metrics/image_only_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("Image-only baseline training completed.")


if __name__ == "__main__":
    main()
