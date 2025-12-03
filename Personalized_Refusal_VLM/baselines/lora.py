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
# 你的配置（100% 按你给的）
# ===============================================
model_name = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
train_json = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"
output_dir = "../../llava_lora_output"

BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 1


# ===============================================
# dataset
# ===============================================
class LLaVADataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f) 
        self.processor = processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]

        # -------- image --------
        image_path = d.get("image", "")
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            img = None

        # -------- conversation --------
        conv = ""
        for c in d["conversations"]:
            import pdb; pdb.set_trace()
            if c["from"] == "human":
                conv += "USER: " + c["value"] + "\n"
            else:
                conv += "ASSISTANT: " + c["value"] + "\n"

        # Processor 会自动构建 input_ids / pixel_values
        batch = self.processor(
            images=img,
            text=conv,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return batch


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

    # -------- LoRA config --------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # llama modules
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

    dataset = LLaVADataset(train_json, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        print(f"===== Epoch {epoch} =====")
        for batch in dataloader:

            # Move to GPU + FP16
            batch = {
                k: v.to(model.device, dtype=torch.float16)
                for k, v in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"],
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