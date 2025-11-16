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
                        #   InstructBlipProcessor, 
                        #   InstructBlipForConditionalGeneration, 
                          AutoModelForCausalLM, 
                          Qwen2_5_VLForConditionalGeneration, 
                          Qwen2VLForConditionalGeneration, 
                          AutoProcessor, 
                          LlavaForConditionalGeneration, 
                          Qwen3VLForConditionalGeneration, 
                          Blip2Processor, 
                          Blip2ForConditionalGeneration,
                          MllamaForConditionalGeneration)

from vti_utils.utils import get_activations_blip, get_activations, get_all_datasets_filter, get_all_datasets
from vti_utils.get_refusal_score import filter_data

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



def process(activations):
    newactivations = [x[0] for x in activations] 
    activations_tensor = torch.stack(newactivations)
    return activations_tensor

def eval_model(args):
    model_path = os.path.join(args.model_path, args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # import pdb; pdb.set_trace()

    if 'llava' in model_path.lower():
        print('Loading Llava model...')
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            # low_cpu_mem_usage=True, 
        )
    elif 'qwen2.5-' in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        )
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
            fp16=True
        ).eval().to(device)
    # elif 'instructblip-' in model_path.lower():
    #     model = InstructBlipForConditionalGeneration.from_pretrained(
    #         model_path, 
    #         device_map="auto", 
    #         torch_dtype=torch.float16,
    #     ).to(device)
    # elif 'blip2-' in model_path.lower():
    #     model = Blip2ForConditionalGeneration.from_pretrained(
    #         model_path, 
    #         device_map="auto", 
    #         torch_dtype=torch.float16,
    #     ).to(device)
    elif 'llama' in model_path.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
        )

    if 'instructblip-' in model_path.lower():
        processor = InstructBlipProcessor.from_pretrained(model_path)
    elif 'blip2-' in model_path.lower():
        processor = Blip2Processor.from_pretrained(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path)

    # Load datasets
    # original_data = get_all_datasets_filter(args)
    # original_data = get_all_datasets(args)
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

    # Filter data based on model scores
    # without_sys_out_train_text, with_sys_out_train_text, out_train_images = filter_data(cfg, model, processor, processor.tokenizer, with_sys_out_train_text, without_sys_out_train_text, out_train_images)

    print('Obtaining direction\n')

    # test_img_tensors = processor(images=out_train_images[0], return_tensors="pt").to('cuda')
    # get_visual_hiddenstates(model, [test_img_tensors], model_is_llaval=False)

    if 'instructblip-' in model_path.lower() or 'blip2-' in model_path.lower():
        with_sys_in_train_activations = process(get_activations_blip(model, with_sys_in_train_text, in_train_images, processor))
        with_sys_out_train_activations = process(get_activations_blip(model, with_sys_out_train_text, out_train_images, processor))
        without_sys_in_train_activations = process(get_activations_blip(model, without_sys_in_train_text, in_train_images, processor))
        without_sys_out_train_activations = process(get_activations_blip(model, without_sys_out_train_text, out_train_images, processor))

        in_test_activations = process(get_activations_blip(model, in_test_text, in_test_images, processor))
        out_test_activations = process(get_activations_blip(model, out_test_text, out_test_images, processor))

    else:
        with_sys_in_train_activations = process(get_activations(cfg, model, with_sys_in_train_text, in_train_images, processor, system_prompt=False))
        import pdb; pdb.set_trace()
        with_sys_out_train_activations = process(get_activations(cfg, model, with_sys_out_train_text, out_train_images, processor, system_prompt=False))
        without_sys_in_train_activations = process(get_activations(cfg, model, without_sys_in_train_text, in_train_images, processor, system_prompt=False))
        without_sys_out_train_activations = process(get_activations(cfg, model, without_sys_out_train_text, out_train_images, processor, system_prompt=False))

        # 1. 加 system prompt 的 others（对应 h_c(Image_{others} + system_prompt)）
        with_sys_image_others_activations = process(
            get_activations(cfg, model, [""] * len(out_train_images), out_train_images, processor, system_prompt=True)
        )

        # 2. 不加 system prompt 的 biology（对应 h_c(Image_{biology} + "None")）
        without_sys_image_biology_activations = process(
            get_activations(cfg, model, [""] * len(in_train_images), in_train_images, processor, system_prompt=False)
        )

        in_test_activations = process(get_activations(cfg, model, in_test_text, in_test_images, processor, system_prompt=False))
        out_test_activations = process(get_activations(cfg, model, out_test_text, out_test_images, processor, system_prompt=False))

    save_path = "../../output/activations/"
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
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()

    config_path = f'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.num_test is not None:
        cfg.num_test = args.num_test
    if args.num_train is not None:  
        cfg.num_train = args.num_train

    set_seed(cfg.seed)
    eval_model(cfg)