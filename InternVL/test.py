import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# ====== 1. 加载模型 ======
path = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

# ====== 2. 加载图像 → pixel_values ======
from utils_internvl2 import load_image   # 你贴的代码最后写成 utils 文件即可

pixel_values = load_image("../jiaxi.jpg", max_num=12).to(torch.bfloat16).cuda()

# ====== 3. 构造 text prompt ======
question = "<image>\nDescribe the image."

# ====== 4. tokenizer （无 images!!!） ======
inputs = tokenizer(question, return_tensors="pt").to(model.device)

# ====== 5. forward ======
with torch.no_grad():
    outputs = model(
        **inputs,                   # input_ids, attention_mask
        pixel_values=pixel_values,  # 显式传图像
        output_hidden_states=True,  # 要求输出 hidden states
        return_dict=True
    )

# ====== 6. 获取 hidden states ======
hidden_states = outputs.hidden_states

print(len(hidden_states))            # #layers + embeddings
print(hidden_states[-1].shape)       # 最后一层 hidden state