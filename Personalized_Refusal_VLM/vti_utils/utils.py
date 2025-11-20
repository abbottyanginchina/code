
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math

# import kornia
from transformers import set_seed

from datasets import load_dataset, concatenate_datasets

import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple

def tensor_to_image(img_tensor):
    array = img_tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)
    
    return image

def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer

def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)


    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor


# def get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=True):
def get_prompts(args, model, tokenizer, data_demos, model_is_llaval=True):
    if model_is_llaval:
        from vti_utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from vti_utils.conversation import conv_templates, SeparatorStyle
        from vti_utils.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        question = ""
        qs_pos = question
        qs_neg = question

        # if hasattr(model.config, 'mm_use_im_start_end'):
        if True:
            use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))

            if use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos

            if use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            # prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            # prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]
            # import pdb; pdb.set_trace()

            prompts_positive  = [conv_pos.get_prompt() + k['revised_question'] for _, k in data_demos.items()]
            prompts_negative  = [conv_neg.get_prompt() + k['original_question'] for _, k in data_demos.items()]
            # import pdb; pdb.set_trace()

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("/root/autodl-tmp/VTI/instructblip-vicuna-7b")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['revised_question'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['original_question'], return_tensors="pt").to(model.device))

        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    else:

        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['h_value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def get_all_datasets(args):
    in_domain = ['physics']
    out_of_domain = ['biology', 'geography', 'writing-strategies', 'figurative-language', 'economics', 'earth-science']
    dataset = load_dataset("../ScienceQA")["train"].filter(lambda e: e["image"] is not None)
    in_train = dataset.filter(lambda example: example["topic"] in in_domain)
    out_train = dataset.filter(lambda example: example["topic"] in out_of_domain)
    sample_in = in_train.shuffle(seed=args.seed).select([i for i in list(range(args.num_train+args.num_test))])
    sample_out = out_train.shuffle(seed=args.seed).select([i for i in list(range(args.num_train+args.num_test))])

    with_sys_in_train_text = []
    with_sys_out_train_text = []
    without_sys_in_train_text = []
    without_sys_out_train_text = []
    in_train_images = []
    out_train_images = []
    in_test_text = []
    out_test_text = []
    in_test_images = []
    out_test_images = []
    sys_prompt = "Say 'I cannot answer' if the question is NOT related to 'physics'. "
    for i in range(args.num_train):
        with_sys_in_train = sys_prompt + sample_in[i]['question']
        with_sys_out_train = sys_prompt + sample_out[i]['question']
        without_sys_in_train_= sample_in[i]['question']
        without_sys_out_train = sample_out[i]['question']
        in_img_train = sample_in[i]['image']
        out_img_train = sample_out[i]['image']

        with_sys_in_train_text.append(with_sys_in_train)
        with_sys_out_train_text.append(with_sys_out_train)
        without_sys_in_train_text.append(without_sys_in_train_)
        without_sys_out_train_text.append(without_sys_out_train)
        in_train_images.append(in_img_train)
        out_train_images.append(out_img_train)

    for i in range(args.num_test):
        in_test= sample_in[i+args.num_train]['question']
        out_test = sample_out[i+args.num_train]['question']
        in_img_test = sample_in[i+args.num_train]['image']
        out_img_test = sample_out[i+args.num_train]['image']

        in_test_text.append(in_test)
        out_test_text.append(out_test)
        in_test_images.append(in_img_test)
        out_test_images.append(out_img_test)

    # return (with_sys_in_train_text, with_sys_out_train_text, without_sys_in_train_text, without_sys_out_train_text, in_train_images, out_train_images, in_test_text, out_test_text, in_test_images, out_test_images)
    original_dataset = {
        "with_sys_in_train_text": with_sys_in_train_text,
        "with_sys_out_train_text": with_sys_out_train_text,
        "without_sys_in_train_text": without_sys_in_train_text,
        "without_sys_out_train_text": without_sys_out_train_text,
        "in_train_images": in_train_images,
        "out_train_images": out_train_images,
        "in_test_text": in_test_text,
        "out_test_text": out_test_text,
        "in_test_images": in_test_images,
        "out_test_images": out_test_images
    }

    return original_dataset

def get_all_datasets_filter(args):
    if args.data.dataset_name == "ScienceQA":
        in_domain = ['physics']
        out_of_domain = ['biology', 'geography', 'writing-strategies', 'figurative-language', 'economics', 'earth-science']
        # dataset = load_dataset(f"{args.model_path}/ScienceQA")["train"].filter(lambda e: e["image"] is not None)
        dataset = load_dataset(f"{args.data.path}/ScienceQA")["train"].filter(lambda e: e["image"] is not None)
        in_train = dataset.filter(lambda example: example["topic"] in in_domain)
        out_train = dataset.filter(lambda example: example["topic"] in out_of_domain)
    elif args.data.dataset_name == "MMMU":
        # --- 定义领域 ---
        in_domain = ['Biology']
        out_of_domain = ['Accounting', 'Psychology', 'Computer_Science', 'Finance', 'Energy_and_Power']
        all_domains = in_domain + out_of_domain

        # --- 分开存储 ---
        all_in_splits, all_out_splits = [], []

        for domain in all_domains:
            try:
                dataset_dict = load_dataset(f"{args.data.path}/MMMU", domain)
                domain_splits = []

                # 拼接 dev / validation / test
                for split in ['dev', 'validation', 'test']:
                    if split in dataset_dict:
                        domain_splits.append(dataset_dict[split])

                if domain_splits:
                    merged = concatenate_datasets(domain_splits)
                    merged = merged.filter(lambda e: e.get("image_1") is not None)  # 保留有图样本

                    # ✅ 根据领域分类存储
                    if domain in in_domain:
                        all_in_splits.append(merged)
                    else:
                        all_out_splits.append(merged)

            except Exception as e:
                print(f"⚠️ Skipped {domain}: {e}")

        # --- 合并各类数据 ---
        in_train = concatenate_datasets(all_in_splits)
        out_train = concatenate_datasets(all_out_splits)

    sample_in = in_train.shuffle(seed=args.seed).select([i for i in list(range(args.num_train+args.num_test))])
    sample_out = out_train.shuffle(seed=args.seed).select([i for i in list(range(min(1500, len(out_train['question']))))])

    with_sys_in_train_text = []
    with_sys_out_train_text = []
    without_sys_in_train_text = []
    without_sys_out_train_text = []
    in_train_images = []
    out_train_images = []
    in_test_text = []
    out_test_text = []
    in_test_images = []
    out_test_images = []
    sys_prompt = f"Say 'I cannot answer' if the question is NOT related to '{in_domain[0]}'. "
    for i in range(args.num_train):
        with_sys_in_train = sys_prompt + sample_in[i]['question']
        without_sys_in_train_= sample_in[i]['question']
        if args.data.dataset_name == "ScienceQA":
            in_img_train = sample_in[i]['image']
        elif args.data.dataset_name == "MMMU":
            in_img_train = sample_in[i]['image_1']
        with_sys_in_train_text.append(with_sys_in_train)
        without_sys_in_train_text.append(without_sys_in_train_)
        in_train_images.append(in_img_train)

    for i in range(args.num_test):
        in_test= sample_in[i+args.num_train]['question']
        out_test = sample_out[i]['question']
        if args.data.dataset_name == "ScienceQA":
            in_img_test = sample_in[i+args.num_train]['image']
            out_img_test = sample_out[i]['image']
        elif args.data.dataset_name == "MMMU":
            in_img_test = sample_in[i+args.num_train]['image_1']
            out_img_test = sample_out[i]['image_1']

        in_test_text.append(in_test)
        out_test_text.append(out_test)
        in_test_images.append(in_img_test)
        out_test_images.append(out_img_test)

    for i in range(len(sample_out['question']) - args.num_test):
        with_sys_out_train = sys_prompt + sample_out[i + args.num_test]['question']
        without_sys_out_train = sample_out[i + args.num_test]['question']
        if args.data.dataset_name == "ScienceQA":
            out_img_train = sample_out[i + args.num_test]['image']
        elif args.data.dataset_name == "MMMU":
            out_img_train = sample_out[i + args.num_test]['image_1']
        with_sys_out_train_text.append(with_sys_out_train)
        without_sys_out_train_text.append(without_sys_out_train)
        out_train_images.append(out_img_train)

    original_dataset = {
        "with_sys_in_train_text": with_sys_in_train_text,
        "with_sys_out_train_text": with_sys_out_train_text,
        "without_sys_in_train_text": without_sys_in_train_text,
        "without_sys_out_train_text": without_sys_out_train_text,
        "in_train_images": in_train_images,
        "out_train_images": out_train_images,
        "in_test_text": in_test_text,
        "out_test_text": out_test_text,
        "in_test_images": in_test_images,
        "out_test_images": out_test_images
    }

    return original_dataset


"""
Using customer dataset
"""
def get_customer_demos(args, image_processor, model, tokenizer, patch_size = 14, file_path="/root/autodl-tmp/VTI/experiments/data/revised_questions.json", model_is_llaval=True):
    oversensitivity_data = load_dataset("./experiments/data/MOSSBench", "oversensitivity")['train'].shuffle(seed=args.seed).select(range(100))
    oversensitivity_img = Image.open(os.path.join('./experiments/data/MOSSBench/images', oversensitivity_data[0]['file_name']))
    oversensitivity_question = oversensitivity_data[0]['question']

    with open(file_path, "r", encoding="utf-8") as f:
        data_demos = json.load(f)

    with open("/root/autodl-tmp/VTI/experiments/data/over_refusal.json", "r", encoding="utf-8") as f:
        over_refusal_data = json.load(f)
        
    # import pdb; pdb.set_trace()

    inputs_images = []
    inputs_texts = []
    for i in range(90):
        # original_question = data_demos[str(i)]['original_question']
        # revised_question = data_demos[str(i)]['revised_question']
        image_path = os.path.join("/root/autodl-tmp/VTI/experiments/data/images", f"{i}.jpg")
        image_raw = Image.open(image_path).convert("RGB")
        # oversensitivity_img = Image.open(os.path.join('./experiments/data/MOSSBench/images', oversensitivity_data[i]['file_name']))
        oversensitivity_img = Image.open(os.path.join(f'./experiments/data/over_refusal/{i}.jpg'))
        oversensitivity_question = oversensitivity_data[i]['question']
        # import pdb; pdb.set_trace()
        # oversensitivity_question = over_refusal_data[str(i)]['text_prompt']

        image_tensor = process_image(image_processor, image_raw)
        inputs_texts.append((oversensitivity_question, data_demos[str(i)]['original_question']))
        inputs_images.append([oversensitivity_img, image_raw])
        # inputs_images.append([image_tensor, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, model_is_llaval=model_is_llaval)
    
    return inputs_images, inputs_texts
    

def get_demos(args, image_processor, model, tokenizer, patch_size = 14, file_path = './experiments/data/hallucination_vti_demos.jsonl', model_is_llaval=True): 
    # Initialize a list to store the JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=model_is_llaval)
    
    return inputs_images, input_ids

def get_activations_qwen(model, inputs_text, image, tokenizer, system_prompt=False):

def get_activations(model, inputs_text, image, processor, system_prompt=False):
    # import pdb; pdb.set_trace()
    h_all = []
    with torch.no_grad():
        for example_id in tqdm(range(len(inputs_text)), desc="Getting activations", total=len(inputs_text)):
            embeddings_for_all_styles = []

            # 兼容：如果只有一个 style（即 inputs_text 是一维列表）
            if isinstance(inputs_text[example_id], str):
                text_list = [inputs_text[example_id]]
            else:
                text_list = inputs_text[example_id]

            for style_id in range(len(text_list)):
                if system_prompt:
                    conversation = [
                        {
                        "role": "system",
                        "content": [
                            {"type": "text", 
                             "text": "Say 'I cannot answer that question.' directly if you refuse to answer."},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text_list[style_id]},
                            ],
                        }
                    ]
                else:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text_list[style_id]},
                            ],
                        }
                    ]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=text, images=image[example_id], return_tensors="pt")

                device = next(model.parameters()).device
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

                with torch.no_grad():
                    h = model(
                        **inputs,
                        output_hidden_states=True,  # 关键参数！
                        return_dict=True
                    )
                    h = h.hidden_states

                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:, -1].detach().cpu())

                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)

            h_all.append(tuple(embeddings_for_all_styles))

    return h_all

def get_activations_blip(model, inputs_text, image, processor):
    h_all = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for example_id in range(len(inputs_text)):
            embeddings_for_all_styles = []

            if isinstance(inputs_text[example_id], str):
                text_list = [inputs_text[example_id]]
            else:
                text_list = inputs_text[example_id]

            for style_id in range(len(text_list)):
                text_query = text_list[style_id]

                inputs = processor(
                    images=image[example_id],
                    text=text_query,
                    return_tensors="pt"
                )

                # 移动到设备
                inputs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                }

                # 前向计算 hidden states
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                # import pdb; pdb.set_trace()
                h = outputs.language_model_outputs.hidden_states

                # 提取每层最后一个 token 的向量
                embedding_token = [
                    h[layer][:, -1].detach().cpu()
                    for layer in range(len(h))
                ]
                embedding_token = torch.cat(embedding_token, dim=0).clone()

                embeddings_for_all_styles.append(embedding_token)

            h_all.append(tuple(embeddings_for_all_styles))

    return h_all

def get_vison_activations_qwen(model, inputs_text, image, processor, system_prompt=False):
    device = next(model.parameters()).device
    h_all = []

    with torch.no_grad():
        for img in image:
            inputs = processor(
                images=img,
                return_tensors="pt"
            ).to(device, torch.float16)

            # ✅ 直接调用视觉部分
            vision_outputs = model.model.visual(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True
            )

            # 提取每层的第 0 个 token
            embedding_token = [
                vision_outputs.hidden_states[layer][:, 0, :].detach().cpu()
                for layer in range(len(vision_outputs.hidden_states))
            ]

            embedding_token = torch.cat(embedding_token, dim=0).clone()
            h_all.append(embedding_token)

    return h_all

def get_hiddenstates(model, inputs_text, image, processor):
        # import pdb; pdb.set_trace()
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs_text)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs_text[example_id])):

                    conversation = [
                        {"role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": inputs_text[example_id][style_id]}
                        ]}
                    ]
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(text=text, images=image[example_id][-1], return_tensors="pt")

                    device = next(model.parameters()).device
                    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

                    with torch.no_grad():
                        h = model(
                            **inputs,
                            output_hidden_states=True,  # 关键参数！
                            return_dict=True
                        ).hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)

                h_all.append(tuple(embeddings_for_all_styles))
    
        return h_all

def obtain_test_vector(model, inputs, image_tensor, rank=1, processor=None):
    """
    Input is one image and text.
    Directly extract the direction from the input hidden states without subtraction.
    """
    hidden_states = get_hiddenstates(model, inputs, image_tensor, processor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data = pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)

    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def obtain_positive_vector(model, inputs, image_tensor, rank=1, processor=None):
    """
    Input is one image and text pair.
    Directly extract the direction from the input hidden states without subtraction.
    """
    hidden_states = get_hiddenstates(model, inputs, image_tensor, processor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    for demonstration_id in range(num_demonstration):
        # Directly use the positive hidden state as the direction
        h = hidden_states[demonstration_id][1].view(-1)
        hidden_states_all.append(h)
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data = pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)

    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][1].size(0), hidden_states[demonstration_id][1].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][1].size(0), hidden_states[demonstration_id][1].size(1))
    return direction, reading_direction

def obtain_negative_vector(model, inputs, image_tensor, rank=1, processor=None):
    """
    Input is one image and text pair.
    Directly extract the direction from the input hidden states without subtraction.
    """
    hidden_states = get_hiddenstates(model, inputs, image_tensor, processor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    for demonstration_id in range(num_demonstration):
        # Directly use the positive hidden state as the direction
        h = hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data = pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)


    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][1].size(0), hidden_states[demonstration_id][1].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][1].size(0), hidden_states[demonstration_id][1].size(1))

    return direction, reading_direction

def obtain_textual_vti(model, inputs, image_tensor, rank=1, processor=None):
    hidden_states = get_hiddenstates(model, inputs, image_tensor, processor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in tqdm(range(num_demonstration), total=num_demonstration, desc="Obtaining textual VTI"):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 

    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor, model_is_llaval=True):
    h_all = []
    with torch.no_grad():
        if model_is_llaval:
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model
        else:
            vision_model = model.model.visual

            inputs = image_tensor[0]
            vision_outputs = vision_model(
                pixel_values=inputs["pixel_values"],
                grid_thw=inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True
            )
            vision_hiddens = vision_outputs.hidden_states


            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True
            
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        if model_is_llaval:
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                        else:
                            _, h_ = vision_model(
                                image_tensor_.unsqueeze(0).cuda())
                        h.append(h_)
                    h = average_tuples(h)
                else:
                    if model_is_llaval:
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states

                    else:
                        _, h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda())
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        if not model_is_llaval:
            model.transformer.visual.output_hidden_states = False

    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1, model_is_llaval=True):

    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llaval = model_is_llaval)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction
