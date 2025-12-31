import torch
import argparse

def load_activations(cfg, layer, output_dir):
    bio_x = torch.load(f"{output_dir}/activations/image_pred_other_layer{layer}_{cfg.model_name}.pt", weights_only=False)[:, layer, :]
    oth_x = torch.load(f"{output_dir}/activations/image_pred_biology_layer{layer}_{cfg.model_name}.pt", weights_only=False)[:, layer, :]
    

    return bio_x, oth_x, bio_target, oth_target