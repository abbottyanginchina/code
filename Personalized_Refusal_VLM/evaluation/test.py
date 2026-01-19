import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks", palette="pastel")


def draw_vision_ablation_grouped_boxplots_dataset(model_name, dataset_name, subjects):
    """
    One figure per dataset:
      x-axis: Subject
      hue: Condition (With/Without)
      y-axis: Refusal Score
    """
    rows = []

    for subject in subjects:
        save_dir_vision = f"../../../results/vision_{model_name}_{dataset_name}_{subject}/refusal_scores"
        save_dir = f"../../../results/output_{model_name}_{dataset_name}_{subject}/refusal_scores"

        with open(f"{save_dir}/out_refusal_scores_{model_name}.pkl", "rb") as f:
            with_scores = pickle.load(f)   # "With" (your original out_refusal_scores)

        with open(f"{save_dir_vision}/vision_out_refusal_scores_{model_name}.pkl", "rb") as f:
            without_scores = pickle.load(f)  # "Without" (vision ablation scores)

        rows += [{"Subject": subject.capitalize(), "Condition": "With", "Refusal Score": s} for s in with_scores]
        rows += [{"Subject": subject.capitalize(), "Condition": "Without", "Refusal Score": s} for s in without_scores]

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5.6))
    ax = plt.gca()

    sns.boxplot(
        data=df,
        x="Subject",
        y="Refusal Score",
        hue="Condition",
        dodge=True,
        width=0.6,
        linewidth=1.2,
        showfliers=False,
        palette=["m", "g"],  # match your reference example
        ax=ax
    )

    # Style
    sns.despine(offset=10, trim=True)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Title mapping
    name_map = {
        "instructblip-vicuna-7b": "InstructBLIP",
        "Idefics3-8B-Llama3": "Idefics",
        "llava-1.5-13b-hf": "Llava-1.5-13B",
        "llava-1.5-7b-hf": "Llava-1.5-7B",
    }
    name_title = name_map.get(model_name, model_name)

    plt.title(f"{name_title} on {dataset_name} (Vision Loss Ablation)",
              fontsize=18, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Refusal Score", fontsize=18)
    plt.xticks(fontsize=22, fontweight="bold")

    # Legend
    leg = ax.legend(title="", loc="upper right", frameon=True)
    for t in leg.get_texts():
        t.set_fontsize(12)

    plt.tight_layout()

    out_path = (
        f"../../../results/vision_ablation_figures/"
        f"vision_ablation_grouped_box_{model_name}_{dataset_name}.png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    model_name = "llava-1.5-7b-hf"

    draw_vision_ablation_grouped_boxplots_dataset(
        model_name=model_name,
        dataset_name="ScienceQA",
        subjects=["biology", "geography", "physics"]
    )

    draw_vision_ablation_grouped_boxplots_dataset(
        model_name=model_name,
        dataset_name="MMMU",
        subjects=["Math", "Art_Theory", "Geography"]
    )