model_id = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"

from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

# model_id = "OpenGVLab/InternVL2-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda().eval()

from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def preprocess_image(img: Image.Image):
    pixel_values = image_transform(img).unsqueeze(0)          # (1, 3, 448, 448)
    image_grid_thw = torch.tensor([[32, 32, 1]], dtype=torch.long)  # grid = 32×32
    image_flags = torch.tensor([[1]], dtype=torch.long)
    return pixel_values, image_grid_thw, image_flags

image = Image.open("../jiaxi.jpg").convert("RGB")
pixel_values, image_grid_thw, image_flags = preprocess_image(image)

text = "Describe this image."
tok = tokenizer(text, return_tensors="pt")