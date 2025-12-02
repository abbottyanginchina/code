import mmengine
import argparse
from vti_utils.utils import get_all_datasets

def generate_answer(cfg):
    original_data = get_all_datasets(cfg)
    import pdb; pdb.set_trace()


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
    config_path = f'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)