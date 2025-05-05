#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to fine-tune InstructPix2Pix for robotic arm frame prediction."""

from types import SimpleNamespace
import logging
import math
import os
from pathlib import Path

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Features
from datasets.features import Value
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# 设置 Hugging Face 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 检查 diffusers 最低版本要求
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    return SimpleNamespace(
        pretrained_model_name_or_path="timbrooks/instruct-pix2pix",
        revision=None,
        train_data_dir="./data/train",
        output_dir=f"./robot_arm_model_scene{os.environ.get('SCENE_ID', '8')}",
        cache_dir=None,
        seed=42,
        resolution=256,
        train_batch_size=32,
        num_train_epochs=100,
        max_train_steps=300,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler="linear",
        lr_warmup_steps=30,
        mixed_precision="fp16",
        report_to="tensorboard",
        checkpointing_steps=50,
        validation_prompt="预测50帧后的机械臂状态",
        num_validation_images=4,
        validation_epochs=1,
        scene_id=int(os.environ.get("SCENE_ID", 8))
    )

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def main():
    args = parse_args()

    # 配置日志和加速器
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(total_limit=5, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # 加载调度器、tokenizer 和模型
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # 冻结 VAE 和文本编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 初始化优化器，仅优化 UNet 参数
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # 定义数据集特征
    features = Features({
        "image": Value("string"),
        "edited_image": Value("string"),
        "edit_prompt": Value("string"),
    })

    # 加载数据集
    train_data_dir = Path(args.train_data_dir) / f"scene{args.scene_id}"
    validation_data_dir = Path(args.train_data_dir.replace("train", "validation")) / f"scene{args.scene_id}"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_data_dir / f"metadata_scene{args.scene_id}.jsonl"),
            "validation": str(validation_data_dir / f"metadata_scene{args.scene_id}_validation.jsonl"),
        },
        features=features,
        cache_dir=args.cache_dir,
    )

    def preprocess_images(examples, data_dir):
        original_images = [convert_to_np(Image.open(os.path.join(data_dir, p)).convert("RGB"), args.resolution) for p in examples["image"]]
        edited_images = [convert_to_np(Image.open(os.path.join(data_dir, p)).convert("RGB"), args.resolution) for p in examples["edited_image"]]
        return np.stack(original_images), np.stack(edited_images)

    def preprocess_split(examples, split_dir):
        original_images, edited_images = preprocess_images(examples, split_dir)
        examples["original_pixel_values"] = torch.tensor(original_images, dtype=torch.float32) / 255.0 * 2 - 1
        examples["edited_pixel_values"] = torch.tensor(edited_images, dtype=torch.float32) / 255.0 * 2 - 1
        examples["input_ids"] = tokenizer(examples["edit_prompt"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return examples

    train_dataset = dataset["train"].with_transform(lambda x: preprocess_split(x, str(train_data_dir)))
    validation_dataset = dataset["validation"].with_transform(lambda x: preprocess_split(x, str(validation_data_dir)))

    def collate_fn(examples):
        return {
            "original_pixel_values": torch.stack([ex["original_pixel_values"] for ex in examples]),
            "edited_pixel_values": torch.stack([ex["edited_pixel_values"] for ex in examples]),
            "input_ids": torch.stack([ex["input_ids"] for ex in examples])
        }

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.train_batch_size)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    validation_dataloader = accelerator.prepare(validation_dataloader)

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    logger.info("***** 开始训练 *****")
    logger.info(f"样本数 = {len(train_dataset)}")
    logger.info(f"训练轮数 = {args.num_train_epochs}")
    logger.info(f"每设备批大小 = {args.train_batch_size}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    train_losses, val_losses = [], []

    for epoch in range(args.num_train_epochs):
        unet.train()
        total_loss, steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                original_latents = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                model_input = torch.cat([noisy_latents, original_latents], dim=1)
                model_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)

            accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=step)
            total_loss += loss.detach().item()
            steps += 1
            if step >= args.max_train_steps:
                break

        avg_train_loss = total_loss / steps
        train_losses.append(avg_train_loss)


        if epoch % args.validation_epochs == 0:
            unet.eval()
            val_loss, val_steps = 0, 0
            for val_batch in validation_dataloader:
                with torch.no_grad():
                    latents = vae.encode(val_batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(val_batch["input_ids"])[0]
                    original_latents = vae.encode(val_batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                    model_input = torch.cat([noisy_latents, original_latents], dim=1)
                    model_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                val_loss += loss.item()
                val_steps += 1
            val_losses.append(val_loss / val_steps if val_steps else 0)
        else:
            val_losses.append(val_losses[-1] if val_losses else 0)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss}, Val Loss = {val_losses[-1]}")

    if accelerator.is_main_process:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'training_val_loss.png'))
        plt.close()

        

        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
