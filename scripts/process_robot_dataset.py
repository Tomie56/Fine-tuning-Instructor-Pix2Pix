"""
Process robotic dataset from multiple episodes and viewpoints,
generate training pairs for LoRA fine-tuning of InstructPix2Pix.
Adapted for: episode0/front_camera_0.jpg, with random 9:1 train-test split
"""

import os
import random
import re
from PIL import Image
from torchvision import transforms
import argparse

# Define task-to-instruction mapping
TASKS = {
    "block_hammer_beat": "hit the block with the hammer",
    "block_handover": "handover the blocks",
    "blocks_stack_easy": "stack blocks"
}

# 提取帧编号
def extract_frame_number(fname, view):
    pattern = rf"{view}_camera_(\d+)"
    match = re.search(pattern, fname)
    return int(match.group(1)) if match else -1

def process_task(task_dir, task_name, output_dir, frame_gap=50, test_split_ratio=0.1):
    view_types = ["front", "head", "right", "left"]
    all_episodes = sorted([e for e in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, e))])

    # 随机打乱 episode，并按比例划分
    random.shuffle(all_episodes)
    num_test = int(len(all_episodes) * test_split_ratio)
    test_episodes = set(all_episodes[:num_test])
    
    transform = transforms.Compose([
        transforms.Resize((128, 128))
    ])

    for epi_idx, episode in enumerate(all_episodes):
        episode_path = os.path.join(task_dir, episode)
        subdir = "test" if episode in test_episodes else "train"

        for view in view_types:
            frames = [f for f in os.listdir(episode_path)
                      if f.startswith(f"{view}_camera_") and f.endswith(".jpg")]
            frames.sort(key=lambda x: extract_frame_number(x, view))

            if len(frames) <= frame_gap:
                continue

            for i in range(len(frames) - frame_gap):
                input_file = frames[i]
                target_file = frames[i + frame_gap]

                input_path_raw = os.path.join(episode_path, input_file)
                target_path_raw = os.path.join(episode_path, target_file)

                try:
                    input_img = Image.open(input_path_raw).convert("RGB")
                    target_img = Image.open(target_path_raw).convert("RGB")
                except Exception as e:
                    print(f"Error reading images: {e}")
                    continue

                input_img = transform(input_img)
                target_img = transform(target_img)

                frame_idx = extract_frame_number(input_file, view)
                target_idx = extract_frame_number(target_file, view)

                prefix = f"{task_name}_{view}_{episode}_f{frame_idx:04d}_to_f{target_idx:04d}"
                input_name = f"{prefix}_input.jpg"
                target_name = f"{prefix}_target.jpg"

                save_input_path = os.path.join(output_dir, subdir, input_name)
                save_target_path = os.path.join(output_dir, subdir, target_name)
                os.makedirs(os.path.dirname(save_input_path), exist_ok=True)

                input_img.save(save_input_path)
                target_img.save(save_target_path)

                annotation_line = f"{input_name}|{TASKS[task_name]}|{target_name}\n"
                with open(os.path.join(output_dir, f"{subdir}_annotations.txt"), "a") as f:
                    f.write(annotation_line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Directory with raw task folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save processed images and annotations")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    for task in TASKS:
        task_path = os.path.join(args.source_dir, f"{task}_D435_pkl")
        if os.path.isdir(task_path):
            print(f"Processing: {task_path}")
            process_task(task_path, task, args.output_dir)
        else:
            print(f"Warning: Directory not found - {task_path}")

if __name__ == "__main__":
    main()
