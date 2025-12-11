import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
import mmengine
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (set_seed, 
                          InstructBlipProcessor, 
                          InstructBlipForConditionalGeneration, 
                          AutoModelForCausalLM, 
                          Qwen2_5_VLForConditionalGeneration, 
                          Qwen2VLForConditionalGeneration, 
                          AutoProcessor, 
                          LlavaForConditionalGeneration, 
                          Qwen3VLForConditionalGeneration, 
                          LlavaNextProcessor, 
                          LlavaNextForConditionalGeneration,
                          AutoModelForVision2Seq,
                          AutoModel,
                          AutoTokenizer)

from vti_utils.utils import get_activations_blip, get_activations, get_all_datasets_filter, get_all_datasets, get_activations
from vti_utils.get_refusal_score import filter_data

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def process(activations):
    newactivations = [x[0] for x in activations] 
    activations_tensor = torch.stack(newactivations)
    return activations_tensor
    
def eval_model(cfg):
    model_path = os.path.join(cfg.model_path, cfg.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 'llava-1.5' in model_path.lower():
        print('Loading Llava model...')
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
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
        ).to(device)
    elif 'internvl-chat-' in model_path.lower():
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True).eval().to(device)
    elif 'qwen3-' in model_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        ).eval().to(device)
    elif 'qwen2.5-' in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        ).to(device)
    elif 'qwen2-' in model_path.lower():
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
            fp16=True,
        ).eval().to(device)
    elif 'instructblip-' in model_path.lower():
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16,
        ).to(device)
    elif 'idefics2-' in model_path.lower():
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            output_hidden_states=True,       
            torch_dtype=torch.float16
        ).to(device).eval()

    # Load processor
    if 'llava-v1.6' in model_path.lower():
        processor = LlavaNextProcessor.from_pretrained(model_path)
    if 'qwen-' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)  # 使用默认tokenizer
    else:
        processor = AutoProcessor.from_pretrained(model_path)

    # Load datasets
    if cfg.data.filter_data:
        original_data = get_all_datasets_filter(cfg)
    else:
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

    torch.cuda.empty_cache()

   
    if cfg.data.filter_data:
        # Filter data based on model scores
        without_sys_out_train_text, with_sys_out_train_text, out_train_images = filter_data(cfg, model, processor, processor.tokenizer, with_sys_out_train_text, without_sys_out_train_text, out_train_images)

    print('Obtaining direction\n')

    with torch.no_grad():
        with_sys_in_train_activations = process(get_activations(model, with_sys_in_train_text, in_train_images, processor, system_prompt=True))
        with_sys_out_train_activations = process(get_activations(model, with_sys_out_train_text, out_train_images, processor, system_prompt=True))
        without_sys_in_train_activations = process(get_activations(model, without_sys_in_train_text, in_train_images, processor, system_prompt=False))
        without_sys_out_train_activations = process(get_activations(model, without_sys_out_train_text, out_train_images, processor, system_prompt=False))

        # 1. 加 system prompt 的 others（对应 h_c(Image_{others} + system_prompt)）
        with_sys_image_others_activations = process(
            get_activations(model, [""] * len(out_train_images), out_train_images, processor, system_prompt=True)
        )
        # 2. 不加 system prompt 的 biology（对应 h_c(Image_{biology} + "None")）
        without_sys_image_biology_activations = process(
            get_activations(model, [""] * len(in_train_images), in_train_images, processor, system_prompt=False)
        )

        in_test_activations = process(get_activations(model, in_test_text, in_test_images, processor, system_prompt=False))
        out_test_activations = process(get_activations(model, out_test_text, out_test_images, processor, system_prompt=False))
        image_in_test_activations = process(
            get_activations(model, [""] * len(in_test_images), in_test_images, processor, system_prompt=False)
        )
        image_out_test_activations = process(
            get_activations(model, [""] * len(out_test_images), out_test_images, processor, system_prompt=False)
        )

    save_path = f"../Persona_output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(with_sys_in_train_activations, f"{save_path}/with_sys_in_train_activations_{cfg.model_name}.pt")
    torch.save(with_sys_out_train_activations, f"{save_path}/with_sys_out_train_activations_{cfg.model_name}.pt")
    torch.save(without_sys_in_train_activations, f"{save_path}/without_sys_in_train_activations_{cfg.model_name}.pt")
    torch.save(without_sys_out_train_activations, f"{save_path}/without_sys_out_train_activations_{cfg.model_name}.pt")
    torch.save(in_test_activations, f"{save_path}/in_test_activations_{cfg.model_name}.pt")
    torch.save(out_test_activations, f"{save_path}/out_test_activations_{cfg.model_name}.pt")

    torch.save(with_sys_image_others_activations, f"{save_path}/with_sys_image_others_activations_{cfg.model_name}.pt")
    torch.save(without_sys_image_biology_activations, f"{save_path}/without_sys_image_biology_activations_{cfg.model_name}.pt")
    torch.save(image_in_test_activations, f"{save_path}/image_in_test_activations_{cfg.model_name}.pt")
    torch.save(image_out_test_activations, f"{save_path}/image_out_test_activations_{cfg.model_name}.pt")

    print("Activations saved.")
def parse_args():
    parser = argparse.ArgumentParser(description="Get Activations")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/gpu02home/jmy5701/gpu/models",
        help="Path to the pretrained models",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=200,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=200,
        help="Number of training samples to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/gpu02home/jmy5701/gpu/data",
        help="Path to the pretrained models",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="biology",
        help="Subject to use",
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()

    config_path = f'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.num_test is not None:
        cfg.num_test = args.num_test
    if args.num_train is not None:  
        cfg.num_train = args.num_train
    if args.dataset is not None:
        cfg.data.dataset_name = args.dataset
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.subject is not None:
        cfg.data.subject = args.subject

    set_seed(cfg.seed)
    eval_model(cfg)