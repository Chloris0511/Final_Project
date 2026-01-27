import torch
from pathlib import Path
from transformers import BertTokenizer
import csv

from models.multimodal_attention_model import AttentionFusionModel
from config.label_map import ID2LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 路径配置 =====
TEST_GUID_FILE = "data/raw/test_without_label.txt"
PROCESSED_TEST_ROOT = Path("data/processed/test")
TEXT_DIR = PROCESSED_TEST_ROOT / "texts_cleaned"
IMAGE_DIR = PROCESSED_TEST_ROOT / "images_processed"

MODEL_PATH = "outputs/checkpoints/attention_fusion.pt"
OUTPUT_FILE = "outputs/test_predictions.csv"

BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128


def load_test_guids(path):
    guids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 跳过表头（如 guid,tag）
            if line.lower().startswith("guid"):
                continue

            # 只取第一列
            guid = line.split(",")[0].strip()
            if guid:
                guids.append(guid)

    return guids

def main():
    print("===== 开始测试集预测 =====")

    # 1. 读取 guid
    guids = load_test_guids(TEST_GUID_FILE)
    print(f"共读取 {len(guids)} 个测试样本")

    # 2. tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 加载模型
    model = AttentionFusionModel(num_labels=len(ID2LABEL))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions = []

    with torch.no_grad():
        for guid in guids:
            # --- 文本 ---
            text_path = TEXT_DIR / f"{guid}.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"缺少文本文件: {text_path}")

            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)

            # --- 图像 ---
            image_path = IMAGE_DIR / f"{guid}.pt"
            if not image_path.exists():
                raise FileNotFoundError(f"缺少图像文件: {image_path}")

            image = torch.load(image_path).unsqueeze(0).to(DEVICE)

            # --- 前向 ---
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image
            )

            pred_id = torch.argmax(logits, dim=1).item()
            pred_label = ID2LABEL[pred_id]

            predictions.append((guid, pred_label))

    # 4. 保存结果
    Path("outputs").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["guid", "label"])
        writer.writerows(predictions)

    print(f"预测完成，结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
