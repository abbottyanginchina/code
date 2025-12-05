import torch
import os
import json
import mmengine
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vti_utils.visualization import visualize_distributions

# ========== 配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"


# import pdb; pdb.set_trace()

class FlowField(nn.Module):
    """
    A model that automatically learns different directions in different input regions.
    Input: x (4096 dimensions)
    Output: x' = x + alpha * p(x) * v(x)
    """
    def __init__(self, input_dim, hidden=1024, ref_vec=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_dim)
        )
        self.gate = nn.Linear(input_dim, 1)            
        self.alpha = nn.Parameter(torch.tensor(2.0))   

    def forward(self, x):
        delta = self.net(x)  
        logits = self.gate(x)            
        p = torch.sigmoid(logits)    
        x_out = x + self.alpha * p * delta 
        return x_out, delta, p, logits

def load_activations(cfg, layer):
    bio_x = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/without_sys_in_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)
    oth_x = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/without_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)
    bio_target = bio_x
    oth_target = torch.load(f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/activations/with_sys_out_train_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device, dtype=torch.float64)

    steering_vec = oth_target.mean(dim=0) - oth_x.mean(dim=0)
    oth_steering = oth_x + steering_vec.unsqueeze(0)

    oth_target = oth_steering

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

# Load training dataset
def train(cfg, start_layer, end_layer):
    output_dir = f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}"
    input_dim = torch.load(f"{output_dir}/activations/without_sys_in_train_activations_{cfg.model_name}.pt", weights_only=False).size(2)

    # === 加载逐样本 ground-truth 激活 ===
    with_sys_image_others = torch.load(f"{output_dir}/activations/with_sys_image_others_activations_{cfg.model_name}.pt", weights_only=False).to(device, dtype=torch.float64)
    without_sys_image_biology = torch.load(f"{output_dir}/activations/without_sys_image_biology_activations_{cfg.model_name}.pt", weights_only=False).to(device, dtype=torch.float64)
    gt_vec = F.normalize(with_sys_image_others - without_sys_image_biology, dim=-1)  # [N, D]


    for layer in range(start_layer, end_layer): 
        print(f"\n===== Training on layer {layer} =====")
        bio_x, oth_x, bio_target, oth_target, steering_vec = load_activations(cfg, layer)
        num_sample = bio_x.size(0)

        # 原始 others 的 ground-truth（只对应 200 个 others）
        gt_vec_layer = gt_vec[:num_sample, layer, :].to(device)
        # ✅ 复制一份，给 biology 占位，拼接成同样长度
        #  biology 的 gt_vec 设为 0 向量，这样在 loss_vision 不会产生额外梯度
        zeros_pad = torch.zeros_like(gt_vec_layer)
        gt_vec_layer = torch.cat([gt_vec_layer, zeros_pad], dim=0).cpu()  # [2 * num_sample, D]

        # Integrate dataset
        x_all = torch.cat([oth_x, bio_x], dim=0).cpu()
        y_all = torch.cat([torch.ones(num_sample), torch.zeros(num_sample)], dim=0).cpu()
        target_all = torch.cat([oth_target, bio_target], dim=0).cpu()

        dataset = torch.utils.data.TensorDataset(x_all, target_all, y_all, gt_vec_layer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)

        model = FlowField(input_dim, ref_vec=steering_vec).to(device, dtype=torch.float64)
        opt = optim.AdamW(model.parameters(), lr=float(cfg.training.lr))

        for epoch in range(cfg.training.epochs):
            total_loss = 0.0
            for x, target, y, gt_vec_batch in data_loader:
                x, target, y, gt_vec_batch = x.to(device), target.to(device), y.to(device), gt_vec_batch.to(device)

                x_pred, delta, p, logits = model(x)

                # masks
                pos_mask = (y == 1)   # others → intervention needed, move closer to target
                neg_mask = (y == 0)   # biology → no intervention, keep as is

                # --- others: direction + reconstruction ---
                if pos_mask.any():      
                    gt_vec_batch = gt_vec_batch.to(device)
                    gt_vec_pos = gt_vec_batch[pos_mask]
                    x_pred_pos = x_pred[pos_mask]
                    target_pos = target[pos_mask]
                    cos_sim = F.cosine_similarity(
                        F.normalize(x_pred_pos, dim=-1),
                        F.normalize(target_pos, dim=-1),
                        dim=-1
                    ).mean()    
                    align = 1- cos_sim  # Minimize direction
                    recon = F.mse_loss(x_pred_pos, target_pos)  # Minimize length

                    # === ✅【新增4】逐样本 vision alignment loss ===
                    # 与 gt_vec_batch 一一对齐，不再用全局平均方向
                    delta_pred = F.normalize(x_pred_pos - x[pos_mask], dim=-1)
                    loss_vision = 1 - F.cosine_similarity(delta_pred, gt_vec_pos, dim=-1).mean()

                    # loss_pos = align + recon + loss_vision
                    loss_pos = align + recon
                else:
                    loss_pos = torch.tensor(0.0, device=device)

                # --- biology: keep as is + penalize intervention ---
                if neg_mask.any():
                    x_neg = x[neg_mask]
                    x_pred_neg = x_pred[neg_mask]
                    p_neg = p[neg_mask]                 # [Bneg,1]
                    keep = F.mse_loss(x_pred_neg, x_neg)  # 尽量不改变
                    sparsity = p_neg.mean()               # 鼓励 p 小
                    mag = (delta[neg_mask]**2).mean()     # 限制改动幅度
                    loss_neg = keep + 0.1 * sparsity + 0.01 * mag
                    loss_neg = keep + 0.1 * sparsity
                else:
                    loss_neg = torch.tensor(0.0, device=device)

                # --- 可选：biology/others 平均方向正交，防塌缩 ---
                delta_bio = delta[pos_mask]
                delta_oth = delta[neg_mask]
                if delta_bio.size(0) > 0 and delta_oth.size(0) > 0:
                    cos_sim = F.cosine_similarity(
                        delta_bio.mean(dim=0, keepdim=True),
                        delta_oth.mean(dim=0, keepdim=True)
                    )
                    loss_ortho = cos_sim.mean() ** 2
                else:
                    loss_ortho = torch.tensor(0.0, device=device)

                # gate 监督
                loss_gate = torch.tensor(0.0, device=device)
                if pos_mask.any():
                    loss_gate += F.binary_cross_entropy_with_logits(logits[pos_mask], torch.ones_like(p[pos_mask]))
                if neg_mask.any():
                    loss_gate += F.binary_cross_entropy_with_logits(logits[neg_mask], torch.zeros_like(p[neg_mask]))

                loss = loss_pos + loss_neg + 1 * loss_ortho + loss_gate

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss {total_loss/len(data_loader):.4f}")

        print(f"✅ Training finished for layer {layer}")

        # Save model
        # save_dir = f"../output_{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}"
        if not os.path.exists(f"{output_dir}/models/"):
            os.makedirs(f"{output_dir}/models/")
        torch.save(model, f"{output_dir}/models/steering_model_layer{layer}_{cfg.model_name}.pt")
        print(f"✅ Saved model for layer {layer}")

        # # ========== 推理阶段 ==========
        # bio_x_test = torch.load(f"{save_dir}/activations/in_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()
        # oth_x_test = torch.load(f"{save_dir}/activations/out_test_activations_{cfg.model_name}.pt", weights_only=False)[:, layer, :].to(device).double()

        # pred_biology, p = infer_dataset(model, bio_x_test, cfg.training.batch_size)
        # # print("Biology intervention probabilities:", p.squeeze().tolist())
        # pred_other,  p = infer_dataset(model, oth_x_test, cfg.training.batch_size)
        # # print("Other intervention probabilities:", p.squeeze().tolist())

        # steering_vec_refusal = pred_other - oth_x_test
        # steering_vec_biology = pred_biology - bio_x_test
        # torch.save(steering_vec_refusal, f"{save_dir}/activations/steering_vec_nonbiology_refusal_layer{layer}_{cfg.model_name}.pt")
        # torch.save(steering_vec_biology, f"{save_dir}/activations/steering_vec_biology_layer{layer}_{cfg.model_name}.pt")


        # save_img = f"{save_dir}/visualizations/"
        # if not os.path.exists(save_img):
        #     os.makedirs(save_img)
        # visualize_distributions(train_other_target=oth_target, train_biology_target=bio_x_test,
        #                         pred_other=pred_other, pred_biology=pred_biology, steered_other=oth_x_test+steering_vec.unsqueeze(0), original_other=oth_x_test,
        #                         save_path=f"{save_img}/activations_{layer}_{cfg.model_name}.png")
        # print(f"✅ Saved steering vectors for layer {layer}")

def parse_args():
    parser = argparse.ArgumentParser(description="Get Activations")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava",
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
        "--subject",
        type=str,
        default="biology",
        help="Subject to use",
    )
   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config_path = 'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.subject is not None:
        cfg.data.subject = args.subject

    train(cfg, args.start_layer, args.end_layer)
