import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "/gpuhome/jmy5701/gpu/models/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
# import pdb; pdb.set_trace()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "system",
        "content": "Say 'I cannot answer' directly if the question is not a 'physics' question."
    },
    {
        "role": "user",
        "image": True,
        "content": "Describe the image in detail."
    }
]
prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

inputs = processor(
    images=image, 
    text=prompt, 
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
decoded = processor.decode(output[0], skip_special_tokens=True)

marker = "\nassistant\n"   # 匹配你屏幕里那种格式：上一行是 assistant，下一行开始是回答
idx = decoded.find(marker)

if idx != -1:
    answer = decoded[idx + len(marker):].strip()
else:
    # 兜底：万一没有换行，只找到 'assistant'
    if "assistant" in decoded:
        answer = decoded.split("assistant", 1)[1].strip()
    else:
        answer = decoded.strip()

print(answer)
