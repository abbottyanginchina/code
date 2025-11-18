from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

model_id = "/gpuhome/jmy5701/gpu/models/Yi-VL-6B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()