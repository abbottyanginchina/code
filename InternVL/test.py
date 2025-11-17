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

# ===== 图像预处理 =====
image = Image.open("../jiaxi.jpg").convert("RGB")
pixel_values = model.process_images([image]).to(model.device)
# shape: [1, 3, 448, 448]

# ===== 文本 token 化 =====
text = "Describe the image."
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# ===== forward（必须显式传 pixel_values）====
with torch.no_grad():
    out = model(
        **inputs,
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = out.hidden_states
print(len(hidden_states))
print(hidden_states[-1].shape)