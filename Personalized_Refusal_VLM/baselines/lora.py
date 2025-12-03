import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model


# ============================
# your config
# ============================
model_name = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
train_json = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"
output_dir = "../../llava_lora_output"

BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 1


# ============================
# Dataset
# ============================
class LLaVADataset(Dataset):
    def __init__(self, json_path, processor):
        self.data = [json.loads(line) for line in open(json_path)]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        image_path = d.get("image", "")

        if image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            img = None

        # Construct dialog
        conv = ""
        for c in d["conversations"]:
            if c["from"] == "human":
                conv += "USER: " + c["value"] + "\n"
            else:
                conv += "ASSISTANT: " + c["value"] + "\n"

        batch = self.processor(
            images=img,
            text=conv,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return batch


# ============================
# Load FP16 LLaVA model
# ============================
def load_model():
    print("Loading LLaVA-1.5 FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,    # FP16!
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # LoRA config — standard for LLAMA/LLaVA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # LLAMA structure
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer, processor


# ============================
# Train
# ============================
def train():
    model, tokenizer, processor = load_model()

    dataset = LLaVADataset(train_json, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Start training...")

    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].cuda().half()

            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("Loss:", loss.item())

    print("Saving LoRA to", output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    train()