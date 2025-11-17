import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

path = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

# ===== 准备输入 =====
text = "Describe the image."
image = Image.open("../jiaxi.jpg").convert("RGB")

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# ===== forward ================
with torch.no_grad():
    out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

# ===== hidden states ===========
hidden_states = out.hidden_states   # list: length = num_layers + embedding
print(len(hidden_states))           # e.g. 33 layers (depends on variant)
print(hidden_states[-1].shape)      # 最后一层 hidden state shape


