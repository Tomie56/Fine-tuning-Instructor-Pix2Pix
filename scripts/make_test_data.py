import os
import json
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 配置路径
input_dir = Path("processed_data")
annotations_file = input_dir / "train_annotations.txt"
image_dir = input_dir / "train"

output_dir = Path("data/test")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取并打乱数据
with open(annotations_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
random.shuffle(lines)

# 随机取前100条
lines = lines[:100]

records = []
for i, line in tqdm(enumerate(lines, start=1), total=100):
    input_name, prompt, target_name = line.split("|")
    input_path = image_dir / input_name
    target_path = image_dir / target_name

    if not input_path.exists() or not target_path.exists():
        print(f"跳过不存在的文件：{input_name} 或 {target_name}")
        continue

    frame_input = f"frame_{i * 2 - 1:05d}.png"
    frame_target = f"frame_{i * 2:05d}.png"

    Image.open(input_path).convert("RGB").save(output_dir / frame_input)
    Image.open(target_path).convert("RGB").save(output_dir / frame_target)

    records.append({
        "image": frame_input,
        "edited_image": frame_target,
        "edit_prompt": f"Predict the state of the arm after 50 frames: {prompt.strip()}"
    })

# 写入 JSONL
jsonl_path = output_dir / "metadata_scene1.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("处理完成！测试数据已保存到 IP2P/data/test/scene1")
