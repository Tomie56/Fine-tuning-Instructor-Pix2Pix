import os
import json
import argparse
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from tqdm import tqdm
from torchvision.transforms import ToTensor, Resize, Compose
import torch


def load_test_data(jsonl_path, image_dir, max_pairs):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f.readlines()[:max_pairs]]
    data = []
    for line in lines:
        data.append({
            "prompt": line["edit_prompt"],
            "input_image": image_dir / line["image"],
            "target_image": image_dir / line["edited_image"]
        })
    return data


def load_pipeline(model_id):
    if model_id == 0:
        model_path = "timbrooks/instruct-pix2pix"
    else:
        model_path = f"IP2P/robot_arm_model_scene{model_id}"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_generation(models, n_samples):
    input_dir = Path("IP2P/data/test")
    jsonl_path = input_dir / "metadata_test.jsonl"
    output_dir = Path("IP2P/sample_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_test_data(jsonl_path, input_dir, n_samples)
    pipelines = {mid: load_pipeline(mid) for mid in models}

    for idx, entry in enumerate(tqdm(data, desc="Generating samples")):
        case_dir = output_dir / f"sample_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        input_image = Image.open(entry["input_image"]).convert("RGB")
        target_image = Image.open(entry["target_image"]).convert("RGB")

        input_image.save(case_dir / "input.png")
        target_image.save(case_dir / "target.png")

        for mid, pipe in pipelines.items():
            out = pipe(prompt=entry["prompt"], image=input_image).images[0]
            out.save(case_dir / f"output_model_{mid}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=int, default=[0, 1, 2, 3], help="Model IDs to use (0=base)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to use (max=100)")
    args = parser.parse_args()

    run_generation(args.models, min(args.num_samples, 100))
