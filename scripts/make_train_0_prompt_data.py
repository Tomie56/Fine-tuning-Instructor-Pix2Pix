import os
import shutil
import json
from pathlib import Path


def copy_and_strip_prompts(src_scene_id=4, dst_scene_id=5):
    base_dir = Path("IP2P/data")
    subsets = ["train", "validation"]

    for subset in subsets:
        src_dir = base_dir / subset / f"scene{src_scene_id}"
        dst_dir = base_dir / subset / f"scene{dst_scene_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片文件
        for file in src_dir.glob("*.png"):
            shutil.copy(file, dst_dir / file.name)

        # 处理 jsonl 文件
        if subset == "train":
            jsonl_filename = f"metadata_scene{src_scene_id}.jsonl"
            new_jsonl_filename = f"metadata_scene{dst_scene_id}.jsonl"
        else:
            jsonl_filename = f"metadata_scene{src_scene_id}_validation.jsonl"
            new_jsonl_filename = f"metadata_scene{dst_scene_id}_validation.jsonl"

        src_jsonl = src_dir / jsonl_filename
        dst_jsonl = dst_dir / new_jsonl_filename

        with open(src_jsonl, "r", encoding="utf-8") as f_in, open(dst_jsonl, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.strip():
                    record = json.loads(line)
                    record["edit_prompt"] = ""
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("复制完成，已清除 edit_prompt 字段。")


if __name__ == "__main__":
    copy_and_strip_prompts()