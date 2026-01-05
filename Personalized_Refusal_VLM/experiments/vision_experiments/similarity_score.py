import torch
import os
import numpy as np
import mmengine
import argparse
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_activations(cfg, layer):
    output_dir = os.path.join(cfg.output_dir, f"output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    image_pred_other_x = torch.load(f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_pred_biology_x = torch.load(f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_in_test_x = torch.load(f"{output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    image_out_test_x = torch.load(f"{output_dir}/activations/ground_truth_image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    out_test_activations = torch.load(f"{output_dir}/activations/out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    in_test_activations = torch.load(f"{output_dir}/activations/in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)

    vision_output_dir = os.path.join(cfg.output_dir, f"vision_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    vision_image_pred_other_x = torch.load(f"{vision_output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    vision_image_pred_biology_x = torch.load(f"{vision_output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    vision_image_in_test_x = torch.load(f"{vision_output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    vision_image_out_test_x = torch.load(f"{vision_output_dir}/activations/ground_truth_image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    
    return (out_test_activations,
            in_test_activations,
            image_pred_other_x, 
            image_pred_biology_x, 
            image_in_test_x, 
            image_out_test_x, 
            vision_image_pred_other_x, 
            vision_image_pred_biology_x, 
            vision_image_in_test_x, 
            vision_image_out_test_x)

def batch_sum_align(pred_shift, true_shift, eps=1e-8, filter_zero=True):
    """
    pred_shift: [N, D] (r'_i)
    true_shift: [N, D] (r_i)
    returns scalar cos(sum r'_i, sum r_i)
    """
    if filter_zero:
        pred_norm = pred_shift.norm(dim=-1)
        true_norm = true_shift.norm(dim=-1)
        mask = (pred_norm > 1e-6) & (true_norm > 1e-6)
        if mask.any():
            pred_shift = pred_shift[mask]
            true_shift = true_shift[mask]

    sum_pred = pred_shift.sum(dim=0)  # [D]
    sum_true = true_shift.sum(dim=0)  # [D]

    # safe cosine
    align = F.cosine_similarity(sum_pred.unsqueeze(0), sum_true.unsqueeze(0), dim=-1, eps=eps)
    return align.item(), int(pred_shift.shape[0])  # also return valid count

def main(cfg):
    with_alignment_scores, without_alignment_scores = [], []
    for layer in range(15, cfg.end_layer - 1):
        (out_test_activations, in_test_activations, image_pred_other_x, image_pred_biology_x, image_in_test_x, image_out_test_x,
         vision_image_pred_other_x, vision_image_pred_biology_x, vision_image_in_test_x, vision_image_out_test_x) = load_activations(cfg, layer)

        # ===== w/ vision loss (your output_ directory) =====
        pred_shift = image_pred_other_x - image_pred_biology_x      # r'_i (pred)
        true_shift = out_test_activations - in_test_activations             # r_i  (gt)
        align_w, n_w = batch_sum_align(pred_shift, true_shift)

        # ===== w/o vision loss (your vision_ directory) =====
        pred_shift_wo = vision_image_pred_other_x - vision_image_pred_biology_x
        # true_shift_wo = vision_image_out_test_x - vision_image_in_test_x
        align_wo, n_wo = batch_sum_align(pred_shift_wo, true_shift)

        # print(f"Layer {layer}: Align(w/ vision-loss)={align_w:.4f} (n={n_w}), Align(w/o vision-loss)={align_wo:.4f} (n={n_wo})")
        without_alignment_scores.append(align_wo)
        with_alignment_scores.append(align_w)
    
    print(f"With alignment scores: {np.mean(with_alignment_scores)}", f"Without alignment scores: {np.mean(without_alignment_scores)}")

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