import torch
from transformers import AutoTokenizer, AutoModel
path = "/gpuhome/jmy5701/gpu/models/InternVL-Chat-V1-5"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
