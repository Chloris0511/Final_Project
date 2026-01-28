import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from models.multimodal_attention_model import AttentionFusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_DIR = Path("data/processed/test/texts_cleaned")
IMAGE_DIR = Path("data/processed/test/images_processed")
TEST_LIST = Path("data/raw/test_without_label.txt")

MODEL_CKPT = "outputs/checkpoints/attention_fusion.pt"
OUT_PATH = "outputs/results/test_predictions.csv"

MAX_LEN = 128

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# 读取 test guid
def load_test_guids(path):
    guids = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头 guid,tag
        for line in f:
            guid = line.strip().split(",")[0]
            guids.append(guid)
    return guids



# 主流程

def main():
    print("===== 开始测试集预测 =====")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 模型
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    image_encoder = resnet18(pretrained=True)
    image_dim = image_encoder.fc.in_features
    image_encoder.fc = nn.Identity()


    model = AttentionFusionModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        text_dim=768,
        image_dim=image_dim,
        hidden_dim=256,
        num_classes=len(ID2LABEL)
    ).to(DEVICE)


    model.load_state_dict(
        torch.load(MODEL_CKPT, map_location=DEVICE)
    )
    model.eval()

    # 读取 guid
    guids = load_test_guids(TEST_LIST)

    results = []

    with torch.no_grad():
        for guid in tqdm(guids, desc="Predicting"):
            text_path = TEXT_DIR / f"{guid}.txt"
            image_path = IMAGE_DIR / f"{guid}.pt"

            if not text_path.exists() or not image_path.exists():
                print(f"[WARN] 缺失数据: {guid}")
                continue

            # 文本
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            # 图像
            image = torch.load(image_path)

            logits = model(
                encoded["input_ids"].to(DEVICE),
                encoded["attention_mask"].to(DEVICE),
                image.unsqueeze(0).to(DEVICE),
                ablation=None
            )

            pred = torch.argmax(logits, dim=1).item()
            results.append([guid, ID2LABEL[pred]])

    # 保存
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["guid", "label"])
        writer.writerows(results)

    print(f"预测完成，保存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
