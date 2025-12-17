import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import set_seed, AutoModelForVision2Seq, LlavaNextProcessor, LlavaNextForConditionalGeneration, InstructBlipForConditionalGeneration, InstructBlipProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration, CLIPImageProcessor, LlavaOnevisionForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
from vti_utils.utils import get_demos, get_customer_demos, obtain_textual_vti, obtain_visual_vti, obtain_negative_vector, obtain_positive_vector, obtain_test_vector, get_all_datasets_filter, get_all_datasets
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers, add_one_layer, remove_one_layer, add_multiple_layers, remove_multiple_layers
from vti_utils.conversation import conv_templates
from vti_utils.SVD import compute_layerwise_V_k

from datasets import load_dataset, concatenate_datasets
import random
import io
import mmengine 

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(img_data):
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    elif isinstance(img_data, np.ndarray):
        return Image.fromarray(img_data).convert("RGB")
    elif isinstance(img_data, str) and os.path.exists(img_data):
        return Image.open(img_data).convert("RGB")
    elif isinstance(img_data, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        raise TypeError(f"无法识别的图片类型: {type(img_data)}")

def eval_model(cfg):
    model_path = os.path.join(cfg.model_path, cfg.model_name)

    if 'llava-1.5' in model_path.lower():
        print("llava model loaded")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(device)
    elif 'llava-onevision-' in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
    elif 'llava-v1.6' in model_path.lower():
        print('Loading Llava model...')
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        )
    elif 'qwen3-' in model_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        ).eval().to(device)
    elif 'qwen2.5-' in model_path.lower():
        print("qwen2.5 model loaded")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        ).to(device)
    elif 'qwen2-' in model_path.lower():
        print("qwen2 model loaded")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
        ).to(device)
    elif 'qwen-' in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            fp16=True
        ).eval().to(device)
    elif 'instructblip-' in model_path.lower():
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16,
        ).to(device)
    elif 'blip2-' in model_path.lower():
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16,
        ).to(device)
    elif 'internvl-chat-' in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            fp16=True
        ).eval().to(device)
    elif 'idefics2-' in model_path.lower():
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            output_hidden_states=True,       
            torch_dtype=torch.float16
        ).to(device).eval()
    elif 'idefics3-' in model_path.lower():
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/Idefics3-8B-Llama3",
            output_hidden_states=True,       
            torch_dtype=torch.float16
        ).to(device).eval()

    if 'qwen-' in model_path.lower():
        processor = QWenProcessor.from_pretrained(model_path)
    elif 'instructblip-' in model_path.lower():
        processor = InstructBlipProcessor.from_pretrained(model_path)
    elif 'blip2-' in model_path.lower():
        processor = Blip2Processor.from_pretrained(model_path)
    elif 'llava-v1.6' in model_path.lower():
        processor = LlavaNextProcessor.from_pretrained(model_path)
    elif 'idefics3-' in model_path.lower():
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
    else:
        processor = AutoProcessor.from_pretrained(model_path)

    print("----------Inference----------")

    # Load datasets

    original_data = get_all_datasets(cfg)
    with_sys_in_train_text = original_data["with_sys_in_train_text"]
    with_sys_out_train_text = original_data["with_sys_out_train_text"]
    without_sys_in_train_text = original_data["without_sys_in_train_text"]
    without_sys_out_train_text = original_data["without_sys_out_train_text"]
    in_train_images = original_data["in_train_images"]
    out_train_images = original_data["out_train_images"]
    in_test_text = original_data["in_test_text"]
    out_test_text = original_data["out_test_text"]
    in_test_images = original_data["in_test_images"]
    out_test_images = original_data["out_test_images"]

    base_path = f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations"
    

    if not os.path.exists(f"../output_{cfg.model_name}_{cfg.data.dataset_name}/results/"):
        os.makedirs(f"../output_{cfg.model_name}_{cfg.data.dataset_name}/results/")

    
    
    # 生成拒绝测试集
    answers_file = f"../output_{cfg.model_name}_{cfg.data.dataset_name}/results/sysprompt_non{cfg.data.subject}_answer_{cfg.model_name}.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img_id in tqdm(range(len(out_test_images))):
        raw_image = load_image(out_test_images[img_id])
        question = out_test_text[img_id]

        torch.cuda.empty_cache()
        if 'instructblip-' in model_path.lower():
            sys_prompt = f"Say 'I cannot answer' if the question is NOT related to '{cfg.data.subject}'. "
            question = sys_prompt + 'USER:'+ question +'\nASSISTANT:'
            inputs = processor(
                    images=raw_image,
                    text=question,
                    return_tensors="pt",
                    truncation=True, 
                    max_length=256,
                ).to(device)
            outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_new_tokens=cfg.max_new_tokens,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        elif 'blip2-' in model_path.lower():
            inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=5,
            )
            answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"Say 'I cannot answer' if the question is NOT related to '{cfg.data.subject}'. "},
                    ],
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
                }, 
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            answer = processor.decode(generated_tokens, skip_special_tokens=True)
        
        assistant_content = answer.split("ASSISTANT")[-1].strip()
        img_save = {
            "model_answer": assistant_content,
            "question": question
        }
        print(answer)
        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
    ans_file.close()


    # 生成in-domain测试集
    answers_file = f"../output_{cfg.model_name}_{cfg.data.dataset_name}/results/sysprompt_{cfg.data.subject}_answer_{cfg.model_name}.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img_id in tqdm(range(len(in_test_images))):
        raw_image = load_image(in_test_images[img_id]) #########
        question = in_test_text[img_id] ##########
        
        torch.cuda.empty_cache()
        if 'instructblip-' in model_path.lower():
            sys_prompt = f"Say 'I cannot answer' if the question is NOT related to '{cfg.data.subject}'. "
            question = sys_prompt + 'USER:'+ question +'\nASSISTANT:'
            inputs = processor(
                    images=raw_image,
                    text=question,
                    return_tensors="pt",
                    truncation=True, 
                    max_length=256,
                ).to(device)
            outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_new_tokens=cfg.max_new_tokens,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        elif 'blip2-' in model_path.lower():
            inputs = processor(raw_image, question, return_tensors="pt").to(device)
            outputs = model.generate(
                    **inputs,
            )
            answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            conversation = [
                {
                "role": "system",
                "content": [
                    {"type": "text", 
                        "text": "Say 'I cannot answer' if the question is NOT related to 'biology'. "},
                    ],
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
                }, 
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            answer = processor.decode(generated_tokens, skip_special_tokens=True)
        
        assistant_content = answer.split("ASSISTANT")[-1].strip()
        img_save = {
            "model_answer": assistant_content,
            "question": question
        }
        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    config_path = 'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to use")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--alpha_image", type=float, default=0)
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--max_layer", type=int, default=33)
    parser.add_argument("--inter_start_layer", type=int, default=15)
    parser.add_argument("--inter_end_layer", type=int, default=32)
    parser.add_argument("--alpha_text", type=float, default=2.)
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--subject", type=str, help="Subject to use")
    args = parser.parse_args()

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.alpha_image is not None:
        cfg.alpha_image = args.alpha_image
    if args.num_train is not None:
        cfg.num_train = args.num_train
    if args.num_test is not None:
        cfg.num_test = args.num_test
    if args.max_layer is not None:
        cfg.max_layer = args.max_layer
    if args.inter_start_layer is not None:
        cfg.inter_start_layer = args.inter_start_layer
    if args.inter_end_layer is not None:
        cfg.inter_end_layer = args.inter_end_layer
    if args.alpha_text is not None:
        cfg.alpha_text = args.alpha_text
    if args.dataset is not None:
        cfg.data.dataset_name = args.dataset
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.subject is not None:
        cfg.data.subject = args.subject

    set_seed(cfg.seed)
    eval_model(cfg)