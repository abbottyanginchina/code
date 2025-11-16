import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"

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

processed = processor(
    images=raw_image,
    text=prompt,
    return_tensors='pt',
    torch_dtype=torch.float16,
)

print("processed dtypes:")
for k, v in processed.items():
    print(k, v.dtype)
import pdb; pdb.set_trace()

inputs = {}
for k, v in processed.items():
    if k == "pixel_values":
        # 图片才转成 fp16
        inputs[k] = v.to("cuda", dtype=torch.float16, non_blocking=True)
    else:
        # input_ids & attention_mask 保持 long / int 类型，不要转 dtype
        inputs[k] = v.to("cuda", non_blocking=True)