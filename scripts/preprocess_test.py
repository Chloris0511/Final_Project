import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = Path("data/raw/data")
TEST_GUID_FILE = Path("data/raw/test_without_label.txt")

OUTPUT_ROOT = Path("data/processed/test")
TEXT_OUT_DIR = OUTPUT_ROOT / "texts_cleaned"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images_processed"

TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 图像处理 =====
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_test_guids(path):
    guids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 跳过表头
            if line.lower().startswith("guid"):
                continue

            # 只取第一列（兼容 csv / txt）
            guid = line.split(",")[0].strip()
            if guid:
                guids.append(guid)

    return guids

def process_text(guid):
    src = RAW_DATA_DIR / f"{guid}.txt"
    dst = TEXT_OUT_DIR / f"{guid}.txt"

    if not src.exists():
        raise FileNotFoundError(f"缺少文本文件: {src}")

    # 和 train 阶段一致的鲁棒读取
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    dst.write_text(text, encoding="utf-8")


def process_image(guid):
    # 兼容 jpg / png
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = RAW_DATA_DIR / f"{guid}{ext}"
        if p.exists():
            img_path = p
            break

    if img_path is None:
        raise FileNotFoundError(f"缺少图像文件: {guid}")

    image = Image.open(img_path).convert("RGB")
    tensor = image_transform(image)

    torch.save(tensor, IMAGE_OUT_DIR / f"{guid}.pt")


def main():
    print("===== 开始 test 数据预处理 =====")
    guids = load_test_guids(TEST_GUID_FILE)
    print(f"共处理 {len(guids)} 个 test 样本")

    for i, guid in enumerate(guids, 1):
        process_text(guid)
        process_image(guid)

        if i % 100 == 0 or i == len(guids):
            print(f"[{i}/{len(guids)}] 已处理")

    print("===== test 数据预处理完成 =====")
    print(f"文本输出目录: {TEXT_OUT_DIR}")
    print(f"图像输出目录: {IMAGE_OUT_DIR}")


if __name__ == "__main__":
    main()
