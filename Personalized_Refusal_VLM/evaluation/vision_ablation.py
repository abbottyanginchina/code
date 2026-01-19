

def draw_vision_ablation(model_name, dataset_name, subject):
    save_dir = f"../../../results/vision_{model_name}_{dataset_name}_{subject}/refusal_scores"

    # Load out_refusal_scores
    with open(f"{save_dir}/vision_out_refusal_scores_{model_name}.pkl", "rb") as f:
        out_refusal_scores = pickle.load(f)


if __name__ == "__main__":
    for model_name in ["llava-1.5-7b-hf"]:
        dataset_name = "ScienceQA"
        for subject in ["biology", "geography", "physics"]:
            draw_vision_ablation(model_name, dataset_name, subject)

        dataset_name = "MMMU"
        for subject in ["Math", "Art_Theory", "Geography"]:
            draw_vision_ablation(model_name, dataset_name, subject)