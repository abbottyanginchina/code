import os
import torch
import mmengine
import argparse

from vti_utils.visualization import visualize_distributions
from vti_utils.nets import FlowField

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_activations(cfg, layer):
    bio_x = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/without_sys_in_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)
    oth_x = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/without_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)
    bio_target = bio_x
    oth_target = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/with_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)

    steering_vec = oth_target.mean(dim=0) - oth_x.mean(dim=0)
    oth_steering = oth_x + steering_vec.unsqueeze(0)

    return bio_x, oth_x, bio_target, oth_target, steering_vec

@torch.no_grad()
def infer(model, x, step_scale=1.0):
    """
    Given a 4096-dimensional vector, outputs x', the direction vector delta, and the gating value p.
    p ≈ the confidence that this sample "requires intervention"; can be used without labels.
    """
    x_pred, delta, p, logits = model(x)
    return x_pred, delta, p

@torch.no_grad()
def infer_dataset(model, X, batch_size=None):
    """
    Perform batch inference on a dataset X.
    Returns:
    X_pred: [N, 4096]
    P:      [N, 1]
    """
    device = next(model.parameters()).device
    N, D = X.shape
    X_pred_list, P_list = [], []
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size].to(device)
        x_pred, delta, p = infer(model, xb)
        X_pred_list.append(x_pred)
        P_list.append(p.cpu())
    X_pred = torch.cat(X_pred_list, dim=0)
    P = torch.cat(P_list, dim=0)
    return X_pred, P

def inference(cfg, model, layer, output_dir):
    # ========== 推理阶段 ==========
    bio_x_test = torch.load(f"{output_dir}/activations/in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()
    oth_x_test = torch.load(f"{output_dir}/activations/out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()
    image_others_x_test = torch.load(f"{output_dir}/activations/image_out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()
    image_biology_x_test = torch.load(f"{output_dir}/activations/image_in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()

    pred_biology, p = infer_dataset(model, bio_x_test, cfg.training.batch_size)
    # print("Biology intervention probabilities:", p.squeeze().tolist())
    pred_other,  p = infer_dataset(model, oth_x_test, cfg.training.batch_size)
    # print("Other intervention probabilities:", p.squeeze().tolist())
    image_pred_other,  p = infer_dataset(model, image_others_x_test, cfg.training.batch_size)
    image_pred_biology, p = infer_dataset(model, image_biology_x_test, cfg.training.batch_size)

    steering_vec_refusal = pred_other - oth_x_test
    steering_vec_biology = pred_biology - bio_x_test
    torch.save(steering_vec_refusal, f"{output_dir}/activations/steering_vec_nonbiology_refusal_layer{layer}_{cfg.model_name}.pt")
    torch.save(steering_vec_biology, f"{output_dir}/activations/steering_vec_biology_layer{layer}_{cfg.model_name}.pt")
    torch.save(image_pred_other, f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt")
    torch.save(image_pred_biology, f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt")

    # if not os.path.exists(cfg.save_dir):
    #     os.makedirs(cfg.save_dir)

    return pred_other, pred_biology, bio_x_test, oth_x_test, steering_vec_refusal
    

def main(cfg):
    output_dir = os.path.join(cfg.output_dir, f"output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}")
    for layer in range(cfg.start_layer, cfg.end_layer): 
        bio_x, oth_x, bio_target, oth_target, steering_vec = load_activations(cfg, layer, output_dir)

        model = FlowField(input_dim=bio_x.shape[1], hidden=1024, ref_vec=steering_vec.to(device)).to(device).double()
        model.load_state_dict(torch.load(f"{output_dir}/models/steering_model_layer{layer}_{cfg.model_name}.pt", weights_only=False).state_dict())
        model.eval()

        pred_other, pred_biology, bio_x_test, oth_x_test, steering_vec_refusal = inference(cfg, model, layer, output_dir)

        visualize_distributions(train_other_target=oth_target, train_biology_target=bio_x_test,
                            pred_other=pred_other, pred_biology=pred_biology, steered_other=oth_x_test+steering_vec.unsqueeze(0), original_other=oth_x_test,
                            save_path=f"{output_dir}/visualizations/activations_{layer}_{cfg.model_name}.png")
        print(f"✅ Saved steering vectors for layer {layer}")

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
    if args.subject is not None:
        cfg.data.subject = args.subject

    main(cfg)