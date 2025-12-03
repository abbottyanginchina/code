import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model


# ===============================================
# config
# ===============================================
model_name = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
train_json = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"
output_dir = "../../llava_lora_output"

BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 1


# ===============================================
# dataset：只返回 img 和 text，不用 processor
# ===============================================
class LLaVADataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        # image
        image_path = d.get("image", "")
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            img = None

        # conversation -> string
        conv = ""
        for c in d["conversations"]:
            if c["from"] == "human":
                conv += "USER: " + c["value"] + "\n"
            else:
                conv += "ASSISTANT: " + c["value"] + "\n"

        return {"image": img, "text": conv}


# ===============================================
# load model (FP16, LLaVA-HF)
# ===============================================
def load_model():
    print("Loading LLaVA-1.5 FP16...")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    # LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # llama attention
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, processor


# ===============================================
# train
# ===============================================
def train():
    model, processor = load_model()

    dataset = LLaVADataset(train_json)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        print(f"===== Epoch {epoch} =====")
        for batch in dataloader:
            images = [item["image"] for item in batch]
            texts = [item["text"] for item in batch]

            # 用 processor 统一构建 input_ids / pixel_values
            inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # 送到 GPU：只把 pixel_values 转成 fp16，token 维持 long
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    model.device, dtype=torch.float16
                )
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(model.device)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values", None),
                labels=inputs["input_ids"],
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print("Loss:", loss.item())

    print("Saving LoRA to", output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    train()