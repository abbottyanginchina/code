from openai import OpenAI
import json
import os
from tqdm import tqdm
import numpy as np

client = OpenAI(api_key="sk-ideottqyxzilborugqgaajfprhyegrcibfyvykxjtyvpyzjm", 
                base_url="https://api.siliconflow.cn/v1")

def chat_LLM(user_prompt):
    sys_prompt = "You are evaluating whether a generated response is linguistically clean. Rate the text on a scale from 0 to 10 based on the presence of garbled or broken language. Directly output the score without any other text."

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {'role': 'user', 
            'content': user_prompt},
            {'role': 'system', 
            'content': sys_prompt}
        ]
    )
    # print("user_prompt: ", user_prompt)

    return response.choices[0].message.content

if __name__ == '__main__':
    # read .jsonl file
    #/home/ubuntu/jiaxi/results/output_llava-1.5-7b-hf_ScienceQA_biology/results/biology_answer_llava-1.5-7b-hf.jsonl
    base_dir = "/home/ubuntu/jiaxi/results"
    model_name = "llava-1.5-7b-hf"
    
    tasks = {
        # "ScienceQA": ["biology", "physics", "geography"],
        "ScienceQA": ["biology"],
        # "MMMU": ["Math", "Art_Theory", "Geography"]
    }
    for dataset, categories in tasks.items():
            for cat in categories:
                #   files = [f"results/biology_answer_{model_name}.jsonl", f"results/nonbiology_answer_{model_name}.jsonl"]
    #             method = "our_method"
    #             output_log_folder = f"/home/ubuntu/jiaxi/LLM_as_judge_results/{dataset}_{cat}"
    #             output_log_file = os.path.join(output_log_folder, f"{model_name}_{method}_answer_quality_results.txt")
    #             if not os.path.exists(output_log_folder):
    #                 os.makedirs(output_log_folder)

    #             with open(output_log_file, 'a', encoding='utf-8') as log_f:
    #                 for file in files:
    #                     data_path = os.path.join(base_dir, f"output_{model_name}_{dataset}_{cat}", file)
    #                     with open(data_path, 'r') as f:
    #                         data_lines = f.readlines()
    #                     data = [json.loads(line) for line in data_lines]
    #                     model_task_info = f"Model: {model_name} | Dataset: {dataset} | Category: {cat} | File: {os.path.basename(file)}"
    #                     log_f.write(f"\n{model_task_info}\n")

    #                     scores = [] 
    #                     for item in tqdm(data, total=len(data)):
    #                         user_response = item['model_answer']
    #                         score = chat_LLM(user_response)

    #                         # Score是否可以转变成float
    #                         try:
    #                             score = float(score)
    #                         except ValueError:
    #                             score = None

    #                         if score is not None:
    #                             print("score: ", score)
    #                             scores.append(score)
                        
    #                     result_str = f"Mean Score: {np.mean(np.array(scores))}, Std Score: {np.std(np.array(scores))}"
    #                     log_f.write(result_str + "\n")
    #                     log_f.flush()
                
                # Baseline method
                method = "sys_prompt"
                files = [f"results/sysprompt_biology_answer_{model_name}.jsonl", f"results/sysprompt_nonbiology_answer_{model_name}.jsonl"]
                output_log_folder = f"/home/ubuntu/jiaxi/LLM_as_judge_results/{dataset}_{cat}"
                output_log_file = os.path.join(output_log_folder, f"{model_name}_{method}_answer_quality_results.txt")
                if not os.path.exists(output_log_folder):
                    os.makedirs(output_log_folder)

                with open(output_log_file, 'a', encoding='utf-8') as log_f:
                    for file in files:
                        data_path = os.path.join(base_dir, f"output_sys_prompt_{model_name}_{dataset}_{cat}", file)
                        with open(data_path, 'r') as f:
                            data_lines = f.readlines()
                        data = [json.loads(line) for line in data_lines]
                        model_task_info = f"Model: {model_name} | Dataset: {dataset} | Category: {cat} | File: {os.path.basename(file)}"
                        log_f.write(f"\n{model_task_info}\n")

                        scores = [] 
                        for item in tqdm(data, total=len(data)):
                            user_response = item['model_answer']
                            score = chat_LLM(user_response)

                            try:
                                score = float(score)
                            except ValueError:
                                score = None

                            if score is not None:
                                scores.append(score)
                        
                        result_str = f"Mean Score: {np.mean(np.array(scores))}, Std Score: {np.std(np.array(scores))}"
                        log_f.write(result_str + "\n")
                        log_f.flush()

    
