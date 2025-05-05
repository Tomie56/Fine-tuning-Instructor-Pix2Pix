import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("/root/autodl-tmp/IP2P/results/scene5_metrics.csv")  # 替换为实际路径

# 定义极端条件
good_mask = (df["ssim"] > 0.95) | (df["psnr"] > 30)
bad_mask = (df["ssim"] < 0.45) &  (df["psnr"] < 5.6)
normal_mask = ~(good_mask | bad_mask)

# 创建图形
plt.figure(figsize=(8, 6))

# 正常样本（蓝色）
plt.scatter(df.loc[normal_mask, "ssim"],
            df.loc[normal_mask, "psnr"],
            alpha=0.6, color='steelblue', label='Normal')

# 极好样本（绿色）
plt.scatter(df.loc[good_mask, "ssim"],
            df.loc[good_mask, "psnr"],
            alpha=0.9, color='green', label='Excellent')

# 极差样本（橙色）
plt.scatter(df.loc[bad_mask, "ssim"],
            df.loc[bad_mask, "psnr"],
            alpha=0.9, color='orange', label='Poor')

# 设置图形样式
plt.xlabel("SSIM")
plt.ylabel("PSNR (dB)")
plt.title("SSIM & PSNR - Model 5")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig("scene5_ssim_psnr_scatter_highlighted_ranges.png")
