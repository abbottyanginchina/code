import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # 注册全局方向（不参与梯度）
        # self.register_buffer("v_ref", ref_vec)

    def forward(self, x):
        delta = self.net(x)  
        # delta = F.normalize(delta, dim=-1)  
        # delta = 0.8 * self.v_ref + 0.2 * delta
        logits = self.gate(x)            
        p = torch.sigmoid(logits)    
        x_out = x + self.alpha * p * delta # x' = x + α·p·v
        return x_out, delta, p, logits