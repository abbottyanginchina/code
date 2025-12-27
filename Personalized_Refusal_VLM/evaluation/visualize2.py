import matplotlib.pyplot as plt
import numpy as np

# 1. 原始数据：(SRS, BRS)
raw_results = {
    "Our Method":    
    [(0.950, 0.05), (0.995, 0), (0.93, 0), (0.97, 0.095), (0.975, 0.015), (0.93, 0.04)],
    "Fine-tuning":   
    [(0.54, 0), (0.09, 0), (0.64, 0), (0.235, 0.02), (0.92, 0), (0.325, 0.01)],
    "Prompt-based":  
    [(0, 0), (0, 0), (0, 0), (0.04, 0.025), (0.065, 0.015), (0.21, 0.32)],
    "Persona":       
    [(0, 0), (0, 0), (0, 0), (0.31, 0.345), (0.335, 0.275), (0.33, 0.44)]
}

# 数据集标签
categories = ["ScienceQA-Bio", "ScienceQA-Phy", "ScienceQA-Geo", "MMMU-Math", "MMMU-Art", "MMMU-Geo"]
n_labels = len(categories)

# 2. 计算 MB-Score 的函数
def get_mb(srs, brs):
    pragmatism = 1 - brs
    if (srs + pragmatism) == 0: return 0
    return 2 * srs * pragmatism / (srs + pragmatism)

# 3. 转换数据：将 (SRS, BRS) 转换为 MB-Score 列表
model_data = {}
for name, results in raw_results.items():
    model_data[name] = [get_mb(s, b) for s, b in results]

# 4. 配置雷达图角度
angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
angles += angles[:1] # 闭合圆圈

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 5. 绘制每个模型
# 建议：第一种颜色（你的方法）使用深蓝色或显眼的颜色
colors = ["#f7a042", "#E270FB", "#2ca02c", "#4b86ce"]
for i, (name, scores) in enumerate(model_data.items()):
    values = scores + scores[:1] # 闭合数据
    ax.plot(angles, values, color=colors[i], linewidth=1., label=name)
    ax.fill(angles, values, color=colors[i], alpha=0.3)

# 6. 装饰与美化
ax.set_theta_offset(np.pi / 2) # 设置起始角度在顶部
ax.set_theta_direction(-1)     # 顺时针排列
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
# --- 核心修改：沿着径向延长线移动标签 ---
labels = ax.get_xticklabels()
for label, angle in zip(labels, angles[:-1]):
    label.set_fontsize(20)
    
    # 获取当前标签的径向位置（默认通常是 1.0 或略大）
    # 通过 set_y 强行将其推到 1.1 或更高。数值越大，离圆周越远
    label.set_y(0.07) # 在极坐标文本中，y 偏移量会沿着半径方向作用
    
    # 自动根据角度调整对齐，防止文字被切断
    ha = 'left' if 0 < angle < np.pi else 'right'
    if angle == 0 or angle == np.pi: ha = 'center'
    label.set_horizontalalignment(ha)

# 如果 set_y 效果不明显（取决于 matplotlib 版本），
# 也可以使用这个更强力的全局参数：
ax.tick_params(axis='x', pad=18) # 这里的 pad 会沿着半径方向推开文字
ax.set_ylim(0, 1.0)            # MB-Score 范围是 0-1

# 添加网格参考线
ax.set_rgrids(np.arange(0.1, 1.1, 0.1), color="black", size=10)

# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
#            ncol=4, frameon=False, fontsize=18)

plt.tight_layout()
plt.show()