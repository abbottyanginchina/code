import mmengine
import base64
import os
import json
import tempfile
from io import BytesIO
from openai import OpenAI
import argparse
from PIL import Image
from tqdm import tqdm
from vti_utils.utils import get_all_datasets

client = OpenAI(api_key="sk-qltonesphqmyxhcnmddxgpncphuneffamlnzzdehyjifwaog", 
                base_url="https://api.siliconflow.cn/v1")


def local_VLM(text, img=None):
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


def pil_to_base64(img: Image.Image, format="PNG"):
    buf = BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"
    
def chat_VLM(text, img=None):
    # img = Image.open(img)
    img = pil_to_base64(img)

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct",
        messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img
                            }
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )

    return response.choices[0].message.content
def generate_answer(cfg):
    original_data = get_all_datasets(cfg)
    without_sys_in_train_text = original_data["without_sys_in_train_text"]
    without_sys_out_train_text = original_data["without_sys_out_train_text"]
    in_train_images = original_data["in_train_images"]
    out_train_images = original_data["out_train_images"]

    results = []
    save_path = f"{cfg.data.path}/{cfg.data.dataset_name}_{cfg.data.subject}_lora"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(f"{save_path}/images")

    for i in tqdm(range(len(without_sys_in_train_text)), total=len(without_sys_in_train_text), desc="Generating in constraint data"):
        text = without_sys_in_train_text[i]
        img = in_train_images[i]
        response = chat_VLM(text, img)
        item = {
            "id": f"train_{i}",
            "image": f"{save_path}/images/{i}.jpg",
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{text}"
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ]
        }
        results.append(item)
        img.save(f"{save_path}/images/{i}.jpg")

    with open(os.path.join(save_path, "test_answer.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for i in tqdm(range(len(without_sys_out_train_text)), total=len(without_sys_out_train_text), desc="Generating out of data"):
        text = without_sys_out_train_text[i]
        img = out_train_images[i]
        response = "Sorry, I cannot answer."
        item = {
            "id": f"train_{i}",
            "image": f"{save_path}/images/{len(without_sys_out_train_text)+i}.jpg",  
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{text}"
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ]
        }
        results.append(item)
        img.save(f"{save_path}/images/{len(without_sys_out_train_text)+i}.jpg")

    with open(os.path.join(save_path, "test_answer.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} items to {save_path}")
        
    print("Finished generating data")

def parse_args():
    parser = argparse.ArgumentParser(description="Lora baseline...")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/gpuhome/jmy5701/gpu/models",
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
        default="/gpuhome/jmy5701/gpu/data",
        help="Path to the pretrained models",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="biology",
        help="Subject to use",
    )
    return parser.parse_args()

if __name__ == '__main__':
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

    generate_answer(cfg)
