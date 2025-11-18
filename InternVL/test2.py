from transformers import AutoModel, AutoProcessor, AutoTokenizer
import torch
from PIL import Image
import torchvision.transforms as transforms

from utils_internvl2 import load_image

model_id = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

internvl2_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def preprocess_internvl2_image(image: Image.Image):
    pixel_values = internvl2_transform(image).unsqueeze(0)
    # Vision Transformer grid: 448/14 = 32 → 32x32 patches
    image_grid_thw = torch.tensor([[32, 32, 1]], dtype=torch.long)  # t = 1 frame
    image_flags = torch.tensor([[1]], dtype=torch.long)
    return pixel_values, image_grid_thw, image_flags

# image = Image.open('../jiaxi.jpg').convert("RGB")
pixel_values = load_image(image_file='../jiaxi.jpg', max_num=12).to(torch.bfloat16).cuda()

image = Image.open('../jiaxi.jpg').convert("RGB")
pixel_values, image_grid_thw, image_flags = preprocess_internvl2_image(image)

# inputs = processor(
#     text="describe this image",
#     return_tensors="pt"
# ).to("cuda")

prompt = "describe this image"
text_inputs = tokenizer(prompt, return_tensors="pt")
inputs = {
    "input_ids": text_inputs["input_ids"].cuda(),
    "attention_mask": text_inputs["attention_mask"].cuda(),
    "pixel_values": pixel_values.half().cuda(),
    # "image_grid_thw": image_grid_thw.cuda(),
    "image_flags": image_flags.cuda(),
}


with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )
import pdb; pdb.set_trace()
hidden_states = outputs.hidden_states  # List of (batch, seq, dim)
print(len(hidden_states))
print(hidden_states[-1].shape)