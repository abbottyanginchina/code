import torch
import os
import mmengine
import argparse
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_activations(cfg, layer):
    output_dir = os.path.join(cfg.output_dir, f"output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    image_pred_other_x = torch.load(f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_pred_biology_x = torch.load(f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    image_in_test_x = torch.load(f"{output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    image_out_test_x = torch.load(f"{output_dir}/activations/image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)

    vision_output_dir = os.path.join(cfg.output_dir, f"vision_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    vision_image_pred_other_x = torch.load(f"{vision_output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    vision_image_pred_biology_x = torch.load(f"{vision_output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False).to(device)
    vision_image_in_test_x = torch.load(f"{vision_output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    vision_image_out_test_x = torch.load(f"{vision_output_dir}/activations/image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device)
    
    return (image_pred_other_x, 
            image_pred_biology_x, 
            image_in_test_x, 
            image_out_test_x, 
            vision_image_pred_other_x, 
            vision_image_pred_biology_x, 
            vision_image_in_test_x, 
            vision_image_out_test_x)

def alignment_metrics(pred_shift, true_shift, eps=1e-8, filter_zero=True):
    """
    pred_shift: [N, D] (r'_i)
    true_shift: [N, D] (r_i)

    Returns:
      mean_cos: mean_i cos(r'_i, r_i)
      std_cos:  std_i  cos(r'_i, r_i)
      dar:      P(cos > 0)
      q05:      5th percentile of cos (tail risk)
      tail10:   mean of bottom 10% cos
      n:        valid sample count
    """
    if filter_zero:
        pred_norm = pred_shift.norm(dim=-1)
        true_norm = true_shift.norm(dim=-1)
        mask = (pred_norm > 1e-6) & (true_norm > 1e-6)
        pred_shift = pred_shift[mask]
        true_shift = true_shift[mask]

    n = int(pred_shift.shape[0])
    if n == 0:
        return {
            "mean_cos": float("nan"),
            "std_cos": float("nan"),
            "dar": float("nan"),
            "q05": float("nan"),
            "tail10": float("nan"),
            "n": 0,
        }

    # safe per-sample cosine
    pred_n = F.normalize(pred_shift, dim=-1, eps=eps)
    true_n = F.normalize(true_shift, dim=-1, eps=eps)
    cos = F.cosine_similarity(pred_n, true_n, dim=-1, eps=eps)  # [n]

    mean_cos = cos.mean().item()
    std_cos = cos.std(unbiased=False).item()
    dar = (cos > 0).float().mean().item()
    q05 = torch.quantile(cos, 0.05).item()

    q10 = torch.quantile(cos, 0.10)
    tail10 = cos[cos <= q10].mean().item()

    return {
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "dar": dar,
        "q05": q05,
        "tail10": tail10,
        "n": n,
    }

def main(cfg):
    for layer in range(15, cfg.end_layer - 1):
        (image_pred_other_x, image_pred_biology_x, image_in_test_x, image_out_test_x,
         vision_image_pred_other_x, vision_image_pred_biology_x, vision_image_in_test_x, vision_image_out_test_x) = load_activations(cfg, layer)

        # ===== w/ vision loss (output_ directory) =====
        pred_shift_w = image_pred_other_x - image_pred_biology_x
        true_shift_w = image_out_test_x - image_in_test_x
        m_w = alignment_metrics(pred_shift_w, true_shift_w)

        # ===== w/o vision loss (vision_ directory) =====
        pred_shift_wo = vision_image_pred_other_x - vision_image_pred_biology_x
        true_shift_wo = vision_image_out_test_x - vision_image_in_test_x
        m_wo = alignment_metrics(pred_shift_wo, true_shift_wo)

        print(
            f"Layer {layer} | "
            f"w/: mean={m_w['mean_cos']:.4f}, dar={m_w['dar']:.3f}, q05={m_w['q05']:.4f}, "
            f"tail10={m_w['tail10']:.4f}, std={m_w['std_cos']:.4f}, n={m_w['n']} | "
            f"w/o: mean={m_wo['mean_cos']:.4f}, dar={m_wo['dar']:.3f}, q05={m_wo['q05']:.4f}, "
            f"tail10={m_wo['tail10']:.4f}, std={m_wo['std_cos']:.4f}, n={m_wo['n']}"
        )

# def main(cfg):
    
#     for layer in range(20, cfg.end_layer + 1):
#         image_pred_other_x, image_pred_biology_x, image_in_test_x, image_out_test_x, vision_image_pred_other_x, vision_image_pred_biology_x, vision_image_in_test_x, vision_image_out_test_x = load_activations(cfg, layer)
        
#         # calculate with vision
#         steering_vec_pred = image_pred_other_x - image_pred_biology_x
#         steering_vec = image_out_test_x - image_in_test_x
#         similarity_score = torch.cosine_similarity(steering_vec_pred, steering_vec, dim=-1)
#         print(f"Layer {layer} similarity score: {similarity_score.mean().item()}")

#         # calculate without vision
#         vision_steering_vec_pred = vision_image_pred_other_x - vision_image_pred_biology_x
#         vision_steering_vec = vision_image_out_test_x - vision_image_in_test_x
#         vision_similarity_score = torch.cosine_similarity(vision_steering_vec_pred, vision_steering_vec, dim=-1)
#         print(f"Layer {layer} vision similarity score: {vision_similarity_score.mean().item()}")
#         import pdb; pdb.set_trace()

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