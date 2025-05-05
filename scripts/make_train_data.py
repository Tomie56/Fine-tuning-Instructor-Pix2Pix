import os
import json
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 配置路径
input_dir = Path("IP2P/processed_data")
annotations_file = input_dir / "train_annotations.txt"
image_dir = input_dir / "train"

output_base = Path("sample_data")
train_dir = output_base / "train" / "scene6"
val_dir = output_base / "validation" / "scene6"

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# 读取注释
with open(annotations_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

random.shuffle(lines)

# 拆分训练集/验证集（默认 9:1 比例）
train_lines = lines[:900]
val_lines = lines[900:1000]

def process_lines(lines, output_dir, jsonl_path, start_idx=1):
    records = []
    for i, line in tqdm(enumerate(lines, start=start_idx), total=len(lines)):
        input_name, prompt, target_name = line.split("|")
        input_path = image_dir / input_name
        target_path = image_dir / target_name

        if not input_path.exists() or not target_path.exists():
            print(f"跳过不存在的文件：{input_name} 或 {target_name}")
            continue

        frame_input = f"frame_{i*2-1:05d}.png"
        frame_target = f"frame_{i*2:05d}.png"

        Image.open(input_path).convert("RGB").save(output_dir / frame_input)
        Image.open(target_path).convert("RGB").save(output_dir / frame_target)

        records.append({
            "image": frame_input,
            "edited_image": frame_target,
            "edit_prompt": f"Predict the state of the arm after 50 frames: {prompt.strip()}"
        })

    # 写 JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 处理训练集
process_lines(train_lines, train_dir, train_dir / "metadata_scene6.jsonl", start_idx=1)

# 处理验证集
process_lines(val_lines, val_dir, val_dir / "metadata_scene6_validation.jsonl", start_idx=101)

print("处理完成！数据已保存到 sample_data/train/scene1 和 sample_data/validation/scene5")
