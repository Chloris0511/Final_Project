import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from datasets.text_dataset import TextDataset
from models.text_model import BertTextClassifier
from utils.metrics import compute_metrics
from config.label_map import LABEL2ID
from config.baseline_config import (
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS
)

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

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(dataloader)

def validate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds

def main():
    # ===== Dataset =====
    train_dataset = TextDataset(
        split_csv="data/processed/train_split.csv",
        processed_root="data/processed/train",
        max_length=MAX_TEXT_LENGTH
    )

    val_dataset = TextDataset(
        split_csv="data/processed/val_split.csv",
        processed_root="data/processed/val",
        max_length=MAX_TEXT_LENGTH
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
    model = BertTextClassifier(
        model_name=BERT_MODEL_NAME,
        num_labels=len(LABEL2ID)
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # ===== Training Loop =====
    print("===== 开始训练 Text-only BERT 模型 =====")

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


    # ===== Save Model =====
    torch.save(model.state_dict(), "outputs/checkpoints/text_only.pt")

    # ===== Save Metrics =====
    with open("outputs/metrics/text_only_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Text-only baseline training completed.")

if __name__ == "__main__":
    main()
