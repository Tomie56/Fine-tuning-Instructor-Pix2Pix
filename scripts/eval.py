import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInstructPix2PixPipeline
import json
import csv


def load_image(path, resolution):
    image = Image.open(path).convert("RGB")
    transform = Compose([
        Resize((resolution, resolution)),
        ToTensor()
    ])
    return transform(image).unsqueeze(0)


def evaluate_model(model_path, data_dir, resolution=256, sample_indices=[0, 1, 2], sample_save_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    json_path = os.path.join(data_dir, "metadata_test.jsonl")
    with open(json_path, "r", encoding="utf-8") as f:
        lines = [eval(line.strip()) for line in f.readlines()][:100]

    ssim_scores, psnr_scores = [], []
    samples_info = []

    from tqdm import tqdm

    for i, sample in enumerate(tqdm(lines, desc=f"Evaluating {model_path}")):
        prompt = sample["edit_prompt"]
        image_path = os.path.join(data_dir, sample["image"])
        gt_path = os.path.join(data_dir, sample["edited_image"])

        input_image = Image.open(image_path).convert("RGB").resize((resolution, resolution))
        gt_image = Image.open(gt_path).convert("RGB").resize((resolution, resolution))

        output = pipe(prompt=prompt, image=input_image).images[0].resize((resolution, resolution))

        output_np = np.array(output).astype(np.float32) / 255.0
        gt_np = np.array(gt_image).astype(np.float32) / 255.0

        ssim = ssim_metric(gt_np, output_np, channel_axis=-1, data_range=1.0)
        psnr = psnr_metric(gt_np, output_np)

        ssim_scores.append(ssim)
        psnr_scores.append(psnr)

        if i in sample_indices and sample_save_dir is not None:
            os.makedirs(sample_save_dir, exist_ok=True)
            input_image.save(os.path.join(sample_save_dir, f"input_{i}.png"))
            output.save(os.path.join(sample_save_dir, f"output_{i}.png"))
            samples_info.append({
                "index": int(i),
                "prompt": prompt,
                "ssim": float(ssim),
                "psnr": float(psnr)
            })

    if sample_save_dir:
        with open(os.path.join(sample_save_dir, "sample_info.json"), "w") as f:
            json.dump(samples_info, f, indent=2, ensure_ascii=False)

    return np.mean(ssim_scores), np.mean(psnr_scores), list(zip(ssim_scores, psnr_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='List of model IDs to evaluate. 0 for base, 1-8 for scene models.')
    args = parser.parse_args()

    base_model = "timbrooks/instruct-pix2pix"
    test_data_root = "IP2P/data/test"
    result_dir = "IP2P/results"
    os.makedirs(result_dir, exist_ok=True)
    resolution = 256
    sample_indices = [0, 1, 2]

    results = {}

    for model_id in args.models:
        if model_id == 0:
            print("Evaluating base model...")
            model_path = base_model
            data_dir = test_data_root
            save_dir = os.path.join(result_dir, "samples", "original")
            name = "base"
        else:
            print(f"Evaluating finetuned model: scene{model_id}")
            model_path = f"IP2P/robot_arm_model_scene{model_id}"
            data_dir = test_data_root
            save_dir = os.path.join(result_dir, "samples", f"scene{model_id}")
            name = f"scene{model_id}"

        ssim, psnr, score_pairs = evaluate_model(
            model_path,
            data_dir,
            resolution,
            sample_indices,
            sample_save_dir=save_dir
        )
        results[name] = {"ssim": float(ssim), "psnr": float(psnr), "pairs": [(float(ss), float(pp)) for ss, pp in score_pairs]}


        # 写入 CSV
        csv_path = os.path.join(result_dir, f"{name}_metrics.csv")
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "ssim", "psnr"])
            for idx, (ss, pp) in enumerate(score_pairs):
                writer.writerow([idx, float(ss), float(pp)])

    # Save results as JSON
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot results
    scenes = list(results.keys())
    ssim_vals = [results[k]["ssim"] for k in scenes]
    psnr_vals = [results[k]["psnr"] for k in scenes]

    x = np.arange(len(scenes))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, ssim_vals, width, label='SSIM')
    ax.bar(x + width/2, psnr_vals, width, label='PSNR')

    ax.set_ylabel('Score')
    ax.set_title('Model Evaluation on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "evaluation_results.png"))
    plt.show()

    # 保存汇总表
    summary_csv = os.path.join(result_dir, "summary_metrics.csv")
    with open(summary_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "avg_ssim", "avg_psnr"])
        for scene in results:
            writer.writerow([scene, float(results[scene]["ssim"]), float(results[scene]["psnr"])])
