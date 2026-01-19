import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_vision_ablation_ci(model_name, dataset_name, subject):
    save_dir_vision = f"../../../results/vision_{model_name}_{dataset_name}_{subject}/refusal_scores"
    save_dir = f"../../../results/output_{model_name}_{dataset_name}_{subject}/refusal_scores"

    with open(f"{save_dir}/out_refusal_scores_{model_name}.pkl", "rb") as f:
        out_scores = np.array(pickle.load(f))
    with open(f"{save_dir_vision}/vision_out_refusal_scores_{model_name}.pkl", "rb") as f:
        no_vision_scores = np.array(pickle.load(f))

    means = [out_scores.mean(), no_vision_scores.mean()]
    cis = [
        1.96 * out_scores.std() / np.sqrt(len(out_scores)),
        1.96 * no_vision_scores.std() / np.sqrt(len(no_vision_scores))
    ]

    plt.figure(figsize=(5.5, 4))
    ax = plt.gca()

    ax.bar(
        ["With Vision Loss", "Without Vision Loss"],
        means,
        yerr=cis,
        capsize=6,
        color=["#6A5ACD", "#B0B0B0"],
        edgecolor="black",
        linewidth=1
    )

    sns.despine(top=True, right=True)
    ax.set_ylabel("Refusal Score", fontsize=14)
    ax.set_title(f"{model_name} ({subject.capitalize()})",
                 fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"../../../results/vision_ablation_figures/"
        f"vision_ablation_ci_{model_name}_{dataset_name}_{subject}.png",
        dpi=160, bbox_inches="tight"
    )

def draw_vision_ablation_violin(model_name, dataset_name, subject):
    save_dir_vision = f"../../../results/vision_{model_name}_{dataset_name}_{subject}/refusal_scores"
    save_dir = f"../../../results/output_{model_name}_{dataset_name}_{subject}/refusal_scores"

    with open(f"{save_dir}/out_refusal_scores_{model_name}.pkl", "rb") as f:
        out_refusal_scores = pickle.load(f)
    with open(f"{save_dir_vision}/vision_out_refusal_scores_{model_name}.pkl", "rb") as f:
        vision_out_refusal_scores = pickle.load(f)

    df = pd.DataFrame({
        "Refusal Score": out_refusal_scores + vision_out_refusal_scores,
        "Type": (["With Vision Loss"] * len(out_refusal_scores)) +
                (["Without Vision Loss"] * len(vision_out_refusal_scores))
    })

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # Violin：分布形态
    sns.violinplot(
        data=df, x="Type", y="Refusal Score",
        inner=None,
        linewidth=0,
        palette=["#6A5ACD", "#B0B0B0"],
        cut=0
    )

    # Box：中位数 + IQR
    sns.boxplot(
        data=df, x="Type", y="Refusal Score",
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1},
        ax=ax
    )

    # 坐标轴样式
    sns.despine(top=True, right=True)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    name_map = {
        "instructblip-vicuna-7b": "InstructBLIP",
        "Idefics3-8B-Llama3": "Idefics",
        "llava-1.5-13b-hf": "LLaVA-1.5-13B",
        "llava-1.5-7b-hf": "LLaVA-1.5-7B"
    }
    name_title = name_map.get(model_name, model_name)

    plt.title(f"{name_title} on {dataset_name} ({subject.capitalize()})",
              fontsize=20, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Refusal Score", fontsize=16)
    plt.xticks(fontsize=15, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"../../../results/vision_ablation_figures/"
        f"vision_ablation_violin_{model_name}_{dataset_name}_{subject}.png",
        dpi=160, bbox_inches="tight"
    )

def draw_vision_ablation(model_name, dataset_name, subject):
    save_dir_vision = f"../../../results/vision_{model_name}_{dataset_name}_{subject}/refusal_scores"
    save_dir = f"../../../results/output_{model_name}_{dataset_name}_{subject}/refusal_scores"

    # Load out_refusal_scores
    with open(f"{save_dir}/out_refusal_scores_{model_name}.pkl", "rb") as f:
        out_refusal_scores = pickle.load(f)

    # Load in_refusal_scores
    with open(f"{save_dir_vision}/vision_out_refusal_scores_{model_name}.pkl", "rb") as f:
        vision_out_refusal_scores = pickle.load(f)

    # 组装 DataFrame
    df = pd.DataFrame({
        "Refusal Score": out_refusal_scores + vision_out_refusal_scores,
        "Type": (["With"] * len(out_refusal_scores)) +
                (["Without"] * len(vision_out_refusal_scores))
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

    # 将title中subject首字母大写
    plt.title(f"{name_title} on {dataset_name} ({subject.capitalize()})", fontsize=22, fontweight="bold")
    plt.xticks(fontsize=22, fontweight="bold")   # x 轴类目（Out-of-constraint / In-constraint）
    # plt.yticks(fontsize=14)   # y 轴刻度
    plt.xlabel("")
    plt.ylabel("Refusal Score", fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{save_dir}/refusal_score_comparison_{model_name}_{dataset_name}_{subject}.pdf", dpi=160, bbox_inches="tight", facecolor='white')
    plt.savefig(f"../../../results/vision_ablation_figures/vision_ablation_refusal_score_comparison_{model_name}_{dataset_name}_{subject}.png", dpi=160, bbox_inches="tight", facecolor='white')



if __name__ == "__main__":
    for model_name in ["llava-1.5-7b-hf"]:
        dataset_name = "ScienceQA"
        for subject in ["biology", "geography", "physics"]:
            draw_vision_ablation_ci(model_name, dataset_name, subject)

        dataset_name = "MMMU"
        for subject in ["Math", "Art_Theory", "Geography"]:
            draw_vision_ablation_ci(model_name, dataset_name, subject)