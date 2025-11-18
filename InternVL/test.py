from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

model_id = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()