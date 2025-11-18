from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_id = "/gpuhome/jmy5701/gpu/models/Yi-VL-6B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()