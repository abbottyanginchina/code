import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np

from vti_utils.SVD import project_onto_svd_subspace


class VTILayer(nn.Module):

    def __init__(self, V_dict, vti_direction, lam):
        super(VTILayer, self).__init__()
        self.vti_direction = vti_direction
        self.lam = lam
        self.V_dict = V_dict

    def forward(self, x):
        if self.vti_direction is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            y = 0
            import pdb; pdb.set_trace()
            for i in range(len(self.vti_direction)):
                if x.size(1) < 2:
                    lambda_sim = 1.0 #+ torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), -self.vti_direction[i][None,None,:], dim=-1)).unsqueeze(-1)
                    clean_vti_direction = project_onto_svd_subspace(self.vti_direction[i], self.V_dict)
                    y += self.lam[i] * lambda_sim * F.normalize(clean_vti_direction, dim=-1).repeat(1,x.shape[1],1)
                else:
                    lambda_sim = 1.0
                    clean_vti_direction = project_onto_svd_subspace(self.vti_direction[i], self.V_dict)
                    y += self.lam[i] * lambda_sim * F.normalize(clean_vti_direction, dim=-1)
            y = y/len(self.vti_direction)
            x = F.normalize(F.normalize(x.float(),dim=-1) +  0.1 * y, dim=-1) * norm
                
            return x.half()
        else:
            return x


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_layers_qwen(model: PreTrainedModel):
    return model.language_model.layers

def get_layers_blip(model: PreTrainedModel):
    return model.language_model.model.layers

def get_layers_blip2(model: PreTrainedModel):
    # import pdb; pdb.set_trace()
    return model.language_model.model.decoder.layers

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_vti_layers(model: PreTrainedModel, vti_drections: Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(vti_drections) == len(layers)
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, VTILayer(vti_drections[i], alpha)) 

def remove_vti_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        vti_mlp = find_module(layer, mlp_keywords)
        layer.mlp = vti_mlp[0]

def add_one_layer(model: PreTrainedModel, vti_drections: Tensor, alpha: list, layer_idx: int):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    layer = layers[layer_idx]
    original_mlp = find_module(layer, mlp_keywords)
    layer.mlp = nn.Sequential(original_mlp, VTILayer(vti_drections, alpha)) 

def remove_one_layer(model: PreTrainedModel, layer_idx: int):
    layers = get_layers(model)
    layer = layers[layer_idx]
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    vti_mlp = find_module(layer, mlp_keywords)
    if isinstance(vti_mlp, nn.Sequential) and len(vti_mlp) == 2 and isinstance(vti_mlp[1], VTILayer):
        layer.mlp = vti_mlp[0]
    else:
        print(f"Layer {layer_idx} 没有 VTILayer 或结构不符合预期")

def add_multiple_layers(V_dict, model: PreTrainedModel, vti_directions: Tensor, alpha: list, layer_indices: list[int], cfg):
    if 'blip2-' in cfg.model_name:
        layers = get_layers_blip2(model)
        for idx in layer_indices:
            layer = layers[idx]
            
            # 保存原始前馈
            original_fc1 = layer.fc1
            original_act = layer.activation_fn
            original_fc2 = layer.fc2

            # 定义新的前馈 forward
            class MLPWithVTI(nn.Module):
                def __init__(self, fc1, act, fc2, vti_dir, alpha):
                    super().__init__()
                    self.fc1 = fc1
                    self.act = act
                    self.fc2 = fc2
                    self.vti = VTILayer(vti_dir, alpha)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.act(x)
                    x = self.fc2(x)
                    x = self.vti(x)
                    return x

            layer.mlp = MLPWithVTI(original_fc1, original_act, original_fc2, vti_directions[idx], alpha)
    else:
        if 'llava' in cfg.model_name.lower():
            layers = get_layers(model)
        elif 'qwen' in cfg.model_name.lower():
            layers = get_layers_qwen(model)
        elif 'instructblip-' in cfg.model_name.lower():
            layers = get_layers_blip(model)
        mlp_keywords = ["mlp", "feedforward", "ffn"]
        # assert len(vti_directions) == len(layers)
        for idx in layer_indices:
            layer = layers[idx]
            original_mlp = find_module(layer, mlp_keywords)
            layer.mlp = nn.Sequential(original_mlp, VTILayer(V_dict[idx], vti_directions[idx], alpha)) 

def remove_multiple_layers(model: PreTrainedModel, layer_indices: list[int], cfg):
    if 'blip2-' in cfg.model_name:
        layers = get_layers_blip2(model)
        for idx in layer_indices:
            layer = layers[idx]
            # 如果layer.mlp是 MLPWithVTI，则直接取原来的 fc1->act->fc2
            if hasattr(layer.mlp, 'fc1') and hasattr(layer.mlp, 'fc2') and hasattr(layer.mlp, 'act'):
                class OriginalMLP(nn.Module):
                    def __init__(self, fc1, act, fc2):
                        super().__init__()
                        self.fc1 = fc1
                        self.act = act
                        self.fc2 = fc2
                    def forward(self, x):
                        x = self.fc1(x)
                        x = self.act(x)
                        x = self.fc2(x)
                        return x
                layer.mlp = OriginalMLP(layer.mlp.fc1, layer.mlp.act, layer.mlp.fc2)
    else:
        if 'llava' in cfg.model_name.lower():
            layers = get_layers(model)
        elif 'qwen' in cfg.model_name.lower():
            layers = get_layers_qwen(model)
        elif 'instructblip-' in cfg.model_name:
            layers = get_layers_blip(model)
        mlp_keywords = ["mlp", "feedforward", "ffn"] 
        for idx in layer_indices:
            layer = layers[idx]
            vti_mlp = find_module(layer, mlp_keywords)
            layer.mlp = vti_mlp[0]