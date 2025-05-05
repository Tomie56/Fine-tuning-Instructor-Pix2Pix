import matplotlib.pyplot as plt

data = {
    "Model 0": {"ssim": 0.6372, "psnr": 11.14},
    "Model 1": {"ssim": 0.6202, "psnr": 10.92},
    "Model 2": {"ssim": 0.6370, "psnr": 11.36},
    "Model 3": {"ssim": 0.6460, "psnr": 11.79},
    "Model 4": {"ssim": 0.6896, "psnr": 12.35},
    "Model 5": {"ssim": 0.5779, "psnr": 8.30},
    "Model 6": {"ssim": 0.6738, "psnr": 11.50},
    "Model 7": {"ssim": 0.7518, "psnr": 16.71},
    "Model 8": {"ssim": 0.6863, "psnr": 12.23},
}

def plot_bar(models, metric, title, filename):
    values = [data[m][metric] for m in models]
    plt.figure(figsize=(6, 4))

    # 使用不同颜色以示区分
    color = 'skyblue' if metric == "ssim" else 'orange'
    plt.bar(models, values, color=color)

    plt.ylabel(metric.upper())
    plt.title(title)

    # 手动设置纵轴范围放大差异
    if metric == "ssim":
        plt.ylim(0.55, 0.77)
    elif metric == "psnr":
        plt.ylim(8, 17.5)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



# 保存图像
plot_bar(["Model 1", "Model 2", "Model 3"], "ssim", "SSIM Comparison (Model 1-3)", "ssim_1_3.png")
plot_bar(["Model 1", "Model 2", "Model 3"], "psnr", "PSNR Comparison (Model 1-3)", "psnr_1_3.png")
plot_bar(["Model 3", "Model 4", "Model 5"], "ssim", "SSIM Comparison (Model 3-5)", "ssim_3_5.png")
plot_bar(["Model 3", "Model 4", "Model 5"], "psnr", "PSNR Comparison (Model 3-5)", "psnr_3_5.png")
plot_bar(["Model 4", "Model 6", "Model 7", "Model 8"], "ssim", "SSIM Comparison (Model 4-8)", "ssim_4_8.png")
plot_bar(["Model 4", "Model 6", "Model 7", "Model 8"], "psnr", "PSNR Comparison (Model 4-8)", "psnr_4_8.png")
