model_id = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"

from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

# model_id = "OpenGVLab/InternVL2-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda().eval()