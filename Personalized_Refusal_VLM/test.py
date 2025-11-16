import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "/gpu02home/jmy5701/gpu/models/llava-1.5-7b-hf"

# 🚀 加载模型：完全由 accelerate 自动分配到 GPU（不要再 .to）
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto",          # 正确
    low_cpu_mem_usage=True,     # 正确
)

processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)

# 🟢 正确的输入处理方式（不会爆显存）
processed = processor(
    images=raw_image,
    text=prompt,
    return_tensors='pt'
)

# 🟩 逐个搬到 GPU（H100 不会碎片/不会 OOM）
inputs = {k: v.to("cuda", dtype=torch.float16, non_blocking=True)
          for k, v in processed.items()}

# 🚀 推理
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# 📝 解码
print(processor.decode(output[0][2:], skip_special_tokens=True))