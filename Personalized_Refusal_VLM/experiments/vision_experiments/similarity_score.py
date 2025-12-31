import torch
import os
import mmengine
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_activations(cfg, layer, output_dir):
    image_pred_other_x = torch.load(f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_pred_biology_x = torch.load(f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_in_test_x = torch.load(f"{output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    image_out_test_x = torch.load(f"{output_dir}/activations/image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    return image_pred_other_x, image_pred_biology_x, image_in_test_x, image_out_test_x

def main(cfg):
    output_dir = os.path.join(cfg.output_dir, f"vision_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    for layer in range(cfg.start_layer, cfg.end_layer + 1):
        image_pred_other_x, image_pred_biology_x, image_in_test_x, image_out_test_x = load_activations(cfg, layer, output_dir)
        print(image_pred_other_x.shape, image_pred_biology_x.shape, image_in_test_x.shape, image_out_test_x.shape)
        import pdb; pdb.set_trace()
        # calculate the similarity score between image_pred_other_x and image_in_test_x
        similarity_score = torch.cosine_similarity(image_pred_other_x, image_in_test_x, dim=-1)
        print(similarity_score)
        # calculate the similarity score between image_pred_biology_x and image_out_test_x
        similarity_score = torch.cosine_similarity(image_pred_biology_x, image_out_test_x, dim=-1)
        print(similarity_score)

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