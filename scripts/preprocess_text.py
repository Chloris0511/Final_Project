# 文本清洗

import csv
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_split(split_name):
    csv_path = PROCESSED_DIR / f"{split_name}_split.csv"
    out_dir = PROCESSED_DIR / split_name / "texts_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            guid = row["guid"]
            text_path = Path(row["text_path"])

            if not text_path.exists():
                continue

            with open(text_path, "r", encoding="utf-8", errors="replace") as tf:
                # 考虑到原始文本存在编码不一致的问题，本文在文本读取阶段采用容错机制，将无法解码的字符统一替换，以保证数据处理流程的稳定性与实验结果的可复现性。
                raw_text = tf.read()

            cleaned = clean_text(raw_text)

            with open(out_dir / f"{guid}.txt", "w", encoding="utf-8") as wf:
                wf.write(cleaned)

            count += 1

    print(f"{split_name} text preprocessing done. Saved {count} files.")

def main():
    process_split("train")
    process_split("val")

if __name__ == "__main__":
    main()
