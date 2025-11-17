import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
import mmengine
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import set_seed, InstructBlipForConditionalGeneration, InstructBlipProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoModel, AutoProcessor, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration, CLIPImageProcessor, LlavaOnevisionForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
from vti_utils.utils import get_demos, get_customer_demos, obtain_textual_vti, obtain_visual_vti, obtain_negative_vector, obtain_positive_vector, obtain_test_vector, get_all_datasets_filter, get_all_datasets
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers, add_one_layer, remove_one_layer, add_multiple_layers, remove_multiple_layers
from vti_utils.conversation import conv_templates

from datasets import load_dataset, concatenate_datasets
import random
import io

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

def eval_model(args):
    model_path = os.path.join(cfg.model_path, cfg.model_name)

    if 'llava' in model_path.lower():
        print("llava model loaded")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(device)
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

    if 'qwen-' in model_path.lower():
        processor = QWenProcessor.from_pretrained(model_path)
    elif 'instructblip-' in model_path.lower():
        processor = InstructBlipProcessor.from_pretrained(model_path)
    elif 'blip2-' in model_path.lower():
        processor = Blip2Processor.from_pretrained(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path)

    print("----------Inference----------")

    # Load datasets
    original_data = get_all_datasets_filter(args)
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

    base_path = "./output/activations"
    steering_list1 = []
    steering_list2 = []
    # llava[1, 33]  qwen[1, 29]

    qwen_max_layer = 28
    llava_max_layer = 33
    blip_max_layer = 33
    max_layer = llava_max_layer if 'llava' in model_path.lower() else qwen_max_layer if 'qwen' in model_path.lower() else blip_max_layer

    # for layer in range(1, args.max_layer): # Must start from 1
    #     path1 = f"{base_path}/steering_vec_nonbiology_refusal_layer{layer}_{cfg.model_name}.pt"
    #     path2 = f"{base_path}/steering_vec_biology_layer{layer}_{cfg.model_name}.pt"
    #     vec1 = torch.load(path1, weights_only=False)
    #     vec2 = torch.load(path2, weights_only=False)
    #     steering_list1.append(vec1)
    #     steering_list2.append(vec2)
    # refusal_all = torch.stack(steering_list1, dim=1)
    # biology_all = torch.stack(steering_list2, dim=1)
    
    # layer = 25
    
    # 这里最大层数是前面减1，因为第零层不取
    target_layers = list(range(args.inter_start_layer, args.inter_end_layer))  # Qwen 1-28  Llava 1-32

    oth_target = torch.load(f"../../output/activations/with_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False).double()
    oth_x = torch.load(f"../../output/activations/without_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False).double()

    refusal_vector = oth_target - oth_x
    # refusal_vector = refusal_vector.mean(dim=0)[1:]

    refusal_vector = oth_target.mean(dim=0) - oth_x.mean(dim=0)
    import pdb; pdb.set_trace()

    # for name, module in model.named_modules():
    #     print(name, type(module))
    
    # 生成拒绝测试集
    if not os.path.exists("../../output/results/"):
        os.makedirs("../../output/results/")

    answers_file = f"./output/results/nonbiology_answer_{cfg.model_name}.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img_id in range(len(out_test_images)):
        raw_image = load_image(out_test_images[img_id])
        question = out_test_text[img_id]
        # add_multiple_layers(model, torch.stack([refusal_all[img_id]],dim=1).cuda(), alpha = [cfg.alpha_text], layer_indices = target_layers, cfg = cfg)
        add_multiple_layers(model, torch.stack([refusal_vector],dim=1).cuda(), alpha = [cfg.alpha_text], layer_indices = target_layers, cfg=cfg)
        # add_multiple_layers(model, steer_vecs, alpha = [cfg.alpha_text], layer_indices = target_layers, cfg=cfg)
        # 只在一层加
        # add_one_layer(model, torch.stack([refusal_vector]).cuda(), alpha = [args.alpha_text], layer_idx = layer)
        # add_one_layer(model, torch.stack([steering_vec_refusal[img_id]]).cuda(), alpha = [args.alpha_text], layer_idx = layer)
        
        torch.cuda.empty_cache()
        if 'instructblip-' in model_path.lower():
            inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
            outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
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
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
                }, 
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            outputs = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            answer = processor.decode(generated_tokens, skip_special_tokens=True)
        print(answer)
        img_save = {
            "model_answer": answer,
            "question": question
        }
        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
        remove_multiple_layers(model, layer_indices = target_layers, cfg = cfg)
        # remove_one_layer(model, layer_idx = layer)
    ans_file.close()

    # 生成in-domain测试集
    answers_file = f"./output/results/biology_answer_{cfg.model_name}.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img_id in range(len(in_test_images)):
        raw_image = load_image(in_test_images[img_id]) #########
        question = in_test_text[img_id] ##########
        add_multiple_layers(model, torch.stack([biology_all[img_id]],dim=1).cuda(), alpha = [cfg.alpha_text], layer_indices = target_layers, cfg = cfg)
        # add_multiple_layers(model, torch.stack([refusal_vector],dim=1).cuda(), alpha = [cfg.alpha_text], layer_indices = target_layers, cfg=cfg)
        # 只在一层加
        # add_one_layer(model, torch.stack([refusal_vector]).cuda(), alpha = [args.alpha_text], layer_idx = layer)
        # add_one_layer(model, torch.stack([steering_vec_refusal[img_id]]).cuda(), alpha = [args.alpha_text], layer_idx = layer)
        
        torch.cuda.empty_cache()
        if 'instructblip-' in model_path.lower():
            inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
            outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
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
            
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            outputs = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            generated_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            answer = processor.decode(generated_tokens, skip_special_tokens=True)
        
        img_save = {
            "model_answer": answer,
            "question": question
        }
        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
        remove_multiple_layers(model, layer_indices = target_layers, cfg = cfg)
        # remove_one_layer(model, layer_idx = layer)
    ans_file.close()

if __name__ == "__main__":
    config_path = 'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava", help="Name of the model to use")
    parser.add_argument("--alpha_image", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--max_layer", type=int, default=29)
    parser.add_argument("--inter_start_layer", type=int, default=15)
    parser.add_argument("--inter_end_layer", type=int, default=32)
    parser.add_argument("--alpha_text", type=float, default=1.7)
    args = parser.parse_args()

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.alpha_image is not None:
        cfg.alpha_image = args.alpha_image
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_train is not None:
        cfg.training.num_train = args.num_train
    if args.num_test is not None:
        cfg.training.num_test = args.num_test
    if args.max_layer is not None:
        cfg.max_layer = args.max_layer
    if args.inter_start_layer is not None:
        cfg.inter_start_layer = args.inter_start_layer
    if args.inter_end_layer is not None:
        cfg.inter_end_layer = args.inter_end_layer
    if args.alpha_text is not None:
        cfg.alpha_text = args.alpha_text

    set_seed(cfg.seed)
    eval_model(cfg)