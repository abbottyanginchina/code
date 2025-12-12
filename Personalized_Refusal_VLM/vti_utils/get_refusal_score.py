import torch
import functools
from contextlib import contextmanager
import argparse
import torch
import os
import io
import json
from tqdm import tqdm
import sys
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
# from transformers import (Qwen2_5_VLForConditionalGeneration, 
#                           AutoModelForCausalLM, 
#                           set_seed, 
#                           AutoTokenizer, AutoModel, AutoProcessor, LlavaForConditionalGeneration, CLIPImageProcessor, LlavaOnevisionForConditionalGeneration

# from vti_utils.utils import get_all_datasets
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers
from vti_utils.conversation import conv_templates

from datasets import load_dataset, concatenate_datasets
import random

def load_image(img_data):
    """
    将不同类型的图片数据统一转换为 PIL.Image.Image RGB。
    img_data 可以是：
        - PIL.Image.Image
        - numpy array
        - 文件路径
        - bytes
    """
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

@contextmanager
def add_hooks(module_forward_pre_hooks, module_forward_hooks, **kwargs):
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(functools.partial(hook, **kwargs)))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(functools.partial(hook, **kwargs)))
        yield
    finally:
        for h in handles:
            h.remove()

def refusal_score(logits, refusal_toks, epsilon=1e-8, tokenizer=None):
    logits = logits.to(torch.float64)
    logits = logits[:, -1, :]  # 只看最后一个token
    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)
    nonrefusal_probs = 1 - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)
    
def get_refusal_scores_llava(
    model,
    processor,
    images,
    conversations,
    tokenizer,
    refusal_toks,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=1
):
    device = model.device
    refusal_scores = torch.zeros(len(conversations), device=device)

    for i in range(0, len(conversations), batch_size):
        batch_conversations = conversations[i:i+batch_size]
        batch_images = images[i:i+batch_size]

        # 1️⃣ 用 chat_template 构造 prompt（自动插入 <image>）
        prompts = [
            processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in batch_conversations
        ]

        # 2️⃣ 构造输入
        inputs = processor(
            images=batch_images,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device, torch.float16)

        # 3️⃣ 前向传播
        with add_hooks(fwd_pre_hooks, fwd_hooks):
            outputs = model(**inputs, output_hidden_states=False, return_dict=True)

        logits = outputs.logits
        refusal_scores[i:i+batch_size] = refusal_score(logits, refusal_toks, tokenizer=tokenizer)

    return refusal_scores

def get_generation_refusal_scores(cfg, model, processor, tokenizer, test_text, test_images):
    scores = []
    for i in range(0, len(test_text)):
        conversation = [
            [
                {"role": "user", "content": [
                {"type": "text", "text": test_text},
                {"type": "image"},
                ]}
            ]
        ]
        refusal_phrases = ["I"]
        refusal_toks = torch.tensor([
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p)[0])
            for p in refusal_phrases
        ], device=model.device)
    
        with torch.no_grad():
            score = get_refusal_scores_llava(
                model,
                processor,
                [test_images],
                conversation,
                tokenizer,
                refusal_toks
            )
        scores.append(score.item())
    return scores

def filter_data(cfg, model, processor, tokenizer, with_sys_out_train_text, without_sys_out_train_text, out_train_images):

    score_with_list, score_no_list, gap_list = [], [], []
    for i in tqdm(range(0, len(with_sys_out_train_text)), desc="Filtering samples", total=len(with_sys_out_train_text)):
        image = load_image(out_train_images[i])
    
        conversations = [
            [
                {"role": "user", "content": [
                    {"type": "text", "text": without_sys_out_train_text[i]},
                    {"type": "image"},
                ]},
            ],
            [
                {"role": "user", "content": [
                    {"type": "text", "text": with_sys_out_train_text[i]},
                    {"type": "image"},
                ]},
            ]
        ]
    
        refusal_phrases = ["I"]
        refusal_toks = torch.tensor([
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p)[0])
            for p in refusal_phrases
        ], device=model.device)
    
        with torch.no_grad():
            scores = get_refusal_scores_llava(
                model,
                processor,
                [image, image],
                conversations,
                tokenizer,
                refusal_toks
            )
        

        score_no_sys, score_with_sys = scores[0].item(), scores[1].item()
        gap = score_with_sys - score_no_sys
        score_with_list.append(score_with_sys)
        score_no_list.append(score_no_sys)
        gap_list.append(gap)
        torch.cuda.empty_cache()

    score_with_tensor = torch.tensor(score_with_list)
    gap_tensor = torch.tensor(gap_list)

    # ✅ 条件筛选：加了 system prompt 后拒绝分数要 > 0
    valid_mask = score_with_tensor > 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze().tolist()

    if isinstance(valid_indices, int):
        valid_indices = [valid_indices]

    if len(valid_indices) == 0:
        print("No samples with positive refusal score after system prompt.")
        return [], [], []

    # 🎲 随机抽样
    k = min(cfg.num_train, len(valid_indices))
    random_indices = random.sample(valid_indices, k=k)

    print(f"\nRandomly selected {k} valid samples (with positive refusal after system prompt):")
    for idx in random_indices:
        print(f"With_sys_score: {score_with_list[idx]:.4f}, No_sys_score: {score_no_list[idx]:.4f}, Index: {idx}")

    selected_with_sys = [with_sys_out_train_text[i] for i in random_indices]
    selected_without_sys = [without_sys_out_train_text[i] for i in random_indices]
    selected_images = [out_train_images[i] for i in random_indices]

    return selected_without_sys, selected_with_sys, selected_images

    # # 取出符合条件的 gap
    # valid_gaps = gap_tensor[valid_indices]

    # # 🔝 按 gap 大小排序取 top-k
    # k = min(cfg.num_train, len(valid_gaps))
    # topk_values, topk_local_indices = torch.topk(valid_gaps, k=k)
    # topk_indices = [valid_indices[i] for i in topk_local_indices.tolist()]

    # print(f"\nSelected top-{k} samples with largest refusal gap (and positive refusal after system prompt):")
    # for v, idx in zip(topk_values.tolist(), topk_indices):
    #     print(f"Gap: {v:.4f}, With_sys_score: {score_with_list[idx]:.4f}, No_sys_score: {score_no_list[idx]:.4f}, Index: {idx}")

    # selected_with_sys = [with_sys_out_train_text[i] for i in topk_indices]
    # selected_without_sys = [without_sys_out_train_text[i] for i in topk_indices]
    # selected_images = [out_train_images[i] for i in topk_indices]

    # return selected_without_sys, selected_with_sys, selected_images

    # mask = scores_tensor < 30.0
    # filtered_scores = scores_tensor[mask]
    # filtered_indices = torch.nonzero(mask, as_tuple=False).squeeze().tolist()

    # if isinstance(filtered_indices, int):
    #     filtered_indices = [filtered_indices]

    # print(f"\nFound {len(filtered_indices)} samples with score < 2.5")

    # if len(filtered_scores) > 1.0:
    #     k = min(cfg.num_train, len(filtered_scores))
    #     topk_values, topk_local_indices = torch.topk(filtered_scores, k=k)

    #     topk_indices = [filtered_indices[i] for i in topk_local_indices.tolist()]

    #     for v, idx in zip(topk_values, topk_indices):
    #         print(f"Score: {v.item():.4f}, Original index: {idx}")

    #     # Select the filtered samples
    #     selected_with_sys = [with_sys_out_train_text[i] for i in topk_indices]
    #     selected_without_sys = [without_sys_out_train_text[i] for i in topk_indices]
    #     selected_images = [out_train_images[i] for i in topk_indices]
    # else:
    #     print("No samples found with score < 2.5")
    #     selected_with_sys, selected_without_sys, selected_images = [], [], []

    # return selected_without_sys, selected_with_sys, selected_images


