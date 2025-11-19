import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
            raise TypeError("Input must be a numpy array or torch tensor.")

def pca_project_2d(datasets):
    """
    datasets: list of [N_i, D] tensors
    返回：
      proj_list: 每个数据投影后的 [N_i, 2]（在 CPU）
      V2: PCA 的投影矩阵 [D, 2]
      mean: 训练的均值 [1, D]
    """
    tensors = [_to_tensor(d) for d in datasets]
    X = torch.cat(tensors, dim=0)          # 合并做共享 PCA
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    U, S, V = torch.pca_lowrank(Xc, q=2)   # 低秩 PCA（稳定，适合高维/小样本）
    V2 = V[:, :2]                           # [D,2]

    proj_list = []
    start = 0
    for t in tensors:
        Tc = t - mean
        proj = Tc @ V2                      # [N_i, 2]
        proj_list.append(proj.cpu())
        start += t.shape[0]
    return proj_list, V2.cpu(), mean.cpu()

def visualize_distributions(train_other_target, train_biology_target,
                            pred_other, pred_biology,
                            title="Activations: Train vs Pred (Other/Biology) via PCA (2D)",
                            save_path=None):
    # 统一投影
    ds = [train_other_target, train_biology_target, pred_other, pred_biology]
    proj_list, V2, mean = pca_project_2d(ds)

    # 画图（matplotlib，单图，不设置颜色）
    plt.figure(figsize=(7, 6))
    labels  = ["Train: other target", "Train: biology target", "Pred: other", "Pred: biology"]
    markers = ["o", "^", "s", "x"]
    for proj, lab, m in zip(proj_list, labels, markers):
        P = proj.detach().cpu().tolist()
        P = np.array(P)
        plt.scatter(P[:, 0], P[:, 1], label=lab, marker=m, alpha=0.6, s=16)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")

def visualize_distributions(
    train_other_target, train_biology_target,
    pred_other, pred_biology,
    steered_other, original_other,
    title="Activations: Train vs Pred vs Steered (Other/Biology) via PCA (2D)",
    save_path=None
):
    # === 统一投影 ===
    ds = [
        train_other_target, train_biology_target,
        pred_other, pred_biology,
        steered_other, original_other
    ]
    proj_list, V2, mean = pca_project_2d(ds)

    # === 绘图 ===
    plt.figure(figsize=(8, 7))
    labels = [
        "Train: other target", "Train: biology target",
        "Pred: other", "Pred: biology",
        "Steered: other", "original: other"
    ]
    markers = ["o", "^", "s", "x", "D", "P"]   # 6 种 marker
    colors  = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    for proj, lab, m, c in zip(proj_list, labels, markers, colors):
        P = proj.detach().cpu().numpy()
        plt.scatter(P[:, 0], P[:, 1], label=lab, marker=m, color=c, alpha=0.6, s=16)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()