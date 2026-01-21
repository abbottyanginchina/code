import matplotlib.pyplot as plt
import numpy as np

# 1. 原始数据 (SRS, BRS)
# 左子图数据 (SQA)
data_sqa = {
    "Our Method":    [(0.91, 0.04), (0.89, 0.05), (0.93, 0.03)], # Bio, Phy, Geo
    "Fine-tuning":   [(0.93, 0.30), (0.92, 0.33), (0.95, 0.28)],
    "Prompt-based":  [(0.73, 0.14), (0.71, 0.15), (0.74, 0.14)],
    "Persona":       [(0.58, 0.07), (0.54, 0.08), (0.56, 0.06)]
}

# 右子图数据 (MMMU)
data_mmmu = {
    "Our Method":    [(0.92, 0.05), (0.88, 0.04), (0.90, 0.06)], # Math, Art, Geo
    "Fine-tuning":   [(0.95, 0.35), (0.94, 0.40), (0.96, 0.38)],
    "Prompt-based":  [(0.75, 0.15), (0.70, 0.12), (0.72, 0.18)],
    "Persona":       [(0.55, 0.08), (0.50, 0.06), (0.52, 0.09)]
}

# 2. 计算 MB-Score
def get_mb(srs, brs):
    pragmatism = 1 - brs
    if (srs + pragmatism) == 0: return 0
    return 2 * srs * pragmatism / (srs + pragmatism)

models = list(data_sqa.keys())
colors = ["#f7a042", "#E270FB", "#2ca02c", "#4b86ce"]
categories_sqa = ["Biology", "Physics", "Geography"]
categories_mmmu = ["Math", "Art Theory", "Geography"]

# 3. 绘图配置
x = np.arange(len(categories_sqa))
width = 0.18 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- 左子图: ScienceQA ---
for i, model in enumerate(models):
    scores = [get_mb(s, b) for s, b in data_sqa[model]]
    pos = x + i*width - (len(models)*width)/2 + width/2
    ax1.bar(pos, scores, width, color=colors[i], edgecolor='black', linewidth=0.8)

ax1.set_title("Science Question Answering (SQA)", fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel("MB-Score", fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories_sqa, fontsize=14, fontweight='bold')

# --- 右子图: MMMU ---
for i, model in enumerate(models):
    scores = [get_mb(s, b) for s, b in data_mmmu[model]]
    pos = x + i*width - (len(models)*width)/2 + width/2
    ax2.bar(pos, scores, width, color=colors[i], edgecolor='black', linewidth=0.8)

ax2.set_title("Massive Multi-discipline (MMMU)", fontsize=16, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(categories_mmmu, fontsize=14, fontweight='bold')

# --- 共有装饰 ---
for ax in [ax1, ax2]:
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2)) # 0.2 为主刻度
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("Combined_Comparison_SQA_MMMU.pdf", bbox_inches='tight')
plt.show()