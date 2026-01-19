




if __name__ == "__main__":
    for model_name in ["llava-1.5-7b-hf"]:
        dataset_name = "ScienceQA"
        for subject in ["biology", "geography", "physics"]:
            draw_04(model_name, dataset_name, subject)

        dataset_name = "MMMU"
        for subject in ["Math", "Art_Theory", "Geography"]:
            draw_04(model_name, dataset_name, subject)