import mmengine
from openai import OpenAI
import argparse
from vti_utils.utils import get_all_datasets

client = OpenAI(api_key="sk-qltonesphqmyxhcnmddxgpncphuneffamlnzzdehyjifwaog", 
                base_url="https://api.siliconflow.cn/v1")
def chat_VLM(user_prompt):

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {'role': 'user', 
            'content': user_prompt},
            {'role': 'system', 
            'content': sys_prompt}
        ]
    )

    return response.choices[0].message.content
def generate_answer(cfg):
    original_data = get_all_datasets(cfg)
    without_sys_in_train_text = original_data["without_sys_in_train_text"]
    import pdb; pdb.set_trace()
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
