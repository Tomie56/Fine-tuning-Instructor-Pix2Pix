import os
import subprocess

# 定义多个实验配置
experiments = [
    {"name": "exp_01", "lr": 2e-5, "steps": 300, "batch": 32, "accum": 8, "scheduler": "cosine", "warmup": 30, "res": 256},
    {"name": "exp_02", "lr": 1e-5, "steps": 300, "batch": 32, "accum": 8, "scheduler": "cosine", "warmup": 30, "res": 256},
    {"name": "exp_03", "lr": 2e-5, "steps": 600, "batch": 32, "accum": 8, "scheduler": "cosine", "warmup": 30, "res": 256},
    {"name": "exp_04", "lr": 2e-5, "steps": 300, "batch": 16, "accum": 4, "scheduler": "cosine", "warmup": 30, "res": 256},
    {"name": "exp_05", "lr": 2e-5, "steps": 300, "batch": 32, "accum": 8, "scheduler": "linear", "warmup": 30, "res": 256},
    {"name": "exp_06", "lr": 2e-5, "steps": 300, "batch": 32, "accum": 8, "scheduler": "cosine", "warmup": 100, "res": 256},
    {"name": "exp_07", "lr": 2e-5, "steps": 300, "batch": 32, "accum": 8, "scheduler": "cosine", "warmup": 30, "res": 384},
]

base_cmd = ["accelerate", "launch", "finetune_instruct_pix2pix.py"]

for i, exp in enumerate(experiments):
    scene_id = 10 + i  # 防止和原始 scene1~6 冲突
    output_dir = f"robot_arm_model_{exp['name']}"

    cmd = base_cmd + [
        "--pretrained_model_name_or_path", "timbrooks/instruct-pix2pix",
        "--train_data_dir", "./data/train",
        "--output_dir", output_dir,
        "--resolution", str(exp["res"]),
        "--train_batch_size", str(exp["batch"]),
        "--gradient_accumulation_steps", str(exp["accum"]),
        "--learning_rate", str(exp["lr"]),
        "--lr_scheduler", exp["scheduler"],
        "--lr_warmup_steps", str(exp["warmup"]),
        "--max_train_steps", str(exp["steps"]),
        "--mixed_precision", "fp16",
        "--seed", "42",
        "--report_to", "tensorboard",
        "--checkpointing_steps", "50",
        "--validation_prompt", "预测50帧后的机械臂状态",
        "--num_validation_images", "4",
        "--validation_epochs", "1",
        "--scene_id", str(scene_id),
    ]

    print(f"Running {exp['name']}...")
    subprocess.run(cmd)
    print(f"Finished {exp['name']}\n")