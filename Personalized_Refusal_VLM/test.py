import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "/gpu02home/jmy5701/gpu/models/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
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

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# 🔥🔥🔥 必须 resize，否则 H100 直接 driver-error oom
raw_image = raw_image.resize((336, 336))

processed = processor(
    images=raw_image,
    text=prompt,
    return_tensors="pt"
)

inputs = {k: v.to("cuda", dtype=torch.float16, non_blocking=True)
          for k, v in processed.items()}

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))