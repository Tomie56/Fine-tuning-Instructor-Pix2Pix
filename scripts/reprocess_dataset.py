import os
import json
import shutil
from pathlib import Path

# 设置基本路径
root_dir = Path("/root/autodl-tmp/IP2P/processed_data")
train_dir = root_dir / "train"
test_dir = root_dir / "test"
scene_id = 1

# 输出目录
scene_train_dir = train_dir / f"scene{scene_id}"
scene_val_dir = test_dir / f"scene{scene_id}"
scene_train_dir.mkdir(parents=True, exist_ok=True)
scene_val_dir.mkdir(parents=True, exist_ok=True)

def convert_and_copy(annotation_path, image_root, output_dir, output_jsonl):
    with open(annotation_path, "r") as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            input_img, prompt, target_img = parts

            # 拷贝图像文件
            for img in [input_img, target_img]:
                src = image_root / img
                dst = output_dir / img
                if not dst.exists():
                    shutil.copy(src, dst)

            # 写入 JSONL 条目
            fout.write(json.dumps({
                "image": input_img,
                "edit_prompt": prompt,
                "edited_image": target_img
            }, ensure_ascii=False) + "\n")

# 处理训练集
train_annotations_txt = train_dir / "train_annotations.txt"
train_jsonl = scene_train_dir / f"metadata_scene{scene_id}.jsonl"
convert_and_copy(train_annotations_txt, train_dir, scene_train_dir, train_jsonl)

# 处理验证集
test_annotations_txt = test_dir / "test_annotations.txt"
val_jsonl = scene_val_dir / f"metadata_scene{scene_id}_validation.jsonl"
convert_and_copy(test_annotations_txt, test_dir, scene_val_dir, val_jsonl)

print("数据转换与图像复制完成！")
