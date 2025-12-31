import torch
import mmengine
import argparse

def load_activations(cfg, layer, output_dir):
    image_pred_other_x = torch.load(f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False)[:, layer, :]
    image_pred_biology_x = torch.load(f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False)[:, layer, :]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Get Activations")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-1.5-7b-hf",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        action="store",
        help="Starting layer for training",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        action="store",
        help="Ending layer for training",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/visualizations/",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ScienceQA",
        help="Dataset to use",
    )
    parser.add_argument(
         "--subject", 
         type=str, 
         default="biology", 
         help="Subject to use"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config_path = 'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.start_layer is not None:
        cfg.start_layer = args.start_layer
    if args.end_layer is not None:
        cfg.end_layer = args.end_layer
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.dataset is not None:
        cfg.data.dataset_name = args.dataset
    if args.subject is not None:
        cfg.data.subject = args.subject

    main(cfg)