import pickle
import os
from seaborn import boxplot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "llava-1.5-7b-hf"
dataset_name = "ScienceQA"
subject = "biology"


def draw_04(model_name, dataset_name, subject):
    save_dir = f"../../../results/output_{model_name}_{dataset_name}_{subject}/refusal_scores"

    # Load out_refusal_scores
    with open(f"{save_dir}/visionout_refusal_scores_{model_name}.pkl", "rb") as f:
        out_refusal_scores = pickle.load(f)

    # Load in_refusal_scores
    with open(f"{save_dir}/vision_in_refusal_scores_{model_name}.pkl", "rb") as f:
        in_refusal_scores = pickle.load(f)

    # 组装 DataFrame
    df = pd.DataFrame({
        "Refusal Score": out_refusal_scores + in_refusal_scores,
        "Type": (["Out-of-constraint"] * len(out_refusal_scores)) +
                (["In-constraint"] * len(in_refusal_scores))
    })

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, x="Type", y="Refusal Score",
        palette=["#6A5ACD", "#FF6F61"],
        showcaps=True, showfliers=False  # 关闭异常值点，可根据需要修改
    )
    sns.swarmplot(  # 若不需要散点可删除这段
        data=df, x="Type", y="Refusal Score",
        color="0.25", alpha=0.7, size=3
    )

    # ===== 外框（坐标轴）样式控制 =====
    ax = plt.gca()

    # 只保留左 / 下边框（推荐）
    # sns.despine(top=True, right=True, left=True, bottom=True)

    # 设置外框颜色和粗细
    ax.spines["left"].set_color("#B0B0B0")    # 浅灰
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["top"].set_color("#B0B0B0")
    ax.spines["right"].set_color("#B0B0B0")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)

    if model_name == "instructblip-vicuna-7b":
        name_title = "InstructBLIP"
    elif model_name == "Idefics3-8B-Llama3":
        name_title = "Idefics"
    elif model_name == "llava-1.5-13b-hf":
        name_title = "Llava-1.5-13B"
    elif model_name == "llava-1.5-7b-hf":
        name_title = "Llava-1.5-7B"
    else:
        name_title = model_name
    plt.title(f"{name_title} on {dataset_name} ({subject})", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=22, fontweight="bold")   # x 轴类目（Out-of-constraint / In-constraint）
    # plt.yticks(fontsize=14)   # y 轴刻度
    plt.xlabel("")
    plt.ylabel("Refusal Score", fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{save_dir}/refusal_score_comparison_{model_name}_{dataset_name}_{subject}.pdf", dpi=160, bbox_inches="tight", facecolor='white')
    plt.savefig(f"../../../refusal_scores_figures/refusal_score_comparison_{model_name}_{dataset_name}_{subject}.pdf", dpi=160, bbox_inches="tight", facecolor='white')


for model_name in ["llava-1.5-7b-hf", "llava-1.5-13b-hf", "Idefics3-8B-Llama3", "instructblip-vicuna-7b"]:
    dataset_name = "ScienceQA"
    for subject in ["biology", "geography", "physics"]:
        draw_04(model_name, dataset_name, subject)

    dataset_name = "MMMU"
    for subject in ["Math", "Art_Theory", "Geography"]:
        draw_04(model_name, dataset_name, subject)

