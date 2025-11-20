import torch

def compute_layerwise_V_k(with_sys, without_sys, k=16):
    """
    with_sys:    [N, L, H] 
    without_sys: [N, L, H]
    
    return:
        V_dict: dict[layer] = V_k  (shape: [H, k])
    """
    N, L, H = with_sys.shape
    V_dict = {}

    for layer in range(L):
        w = with_sys[:, layer, :]        # [N, H]
        x = without_sys[:, layer, :]     # [N, H]
        D = w - x                        # [N, H]

        # SVD on D
        _, _, V = torch.svd(D)           # V: [H, H]
        V_k = V[:, :k]                   # [H, k]

        V_dict[layer] = V_k
    
    return V_dict

def project_onto_svd_subspace(delta_raw, V_k):
    """
    delta_raw: [H]  原始 steering vector（某一层）
    V_k: [H, k]     来自方法1的该层的 SVD 子空间

    return:
        delta_clean: [H]
    """
    # (V_k.T @ delta_raw) → shape [k]
    # V_k @ (…) → shape [H]
    delta_clean = V_k @ (V_k.T @ delta_raw)
    return delta_clean