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

image = Image.open('../jiaxi.jpg').convert("RGB")

inputs = processor(
    image=image,
    text="describe this image",
    return_tensors="pt"
).to("cuda")
import pdb; pdb.set_trace()

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states  # List of (batch, seq, dim)
print(len(hidden_states))
print(hidden_states[-1].shape)