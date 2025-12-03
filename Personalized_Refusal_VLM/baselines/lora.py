import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ===========================
# Config
# ===========================
model_name = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
train_json = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"   # ← 你的数据路径
output_dir = "../../llava_lora_output"

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ====== CONFIG ======
MODEL_NAME = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
DATA_PATH = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"
IMAGE_FOLDER = "images"
OUTPUT_DIR = "../../llava_lora_output"
BATCH_SIZE = 1
LR = 2e-4
NUM_EPOCHS = 1


# ====== Dataset ======
class LLaVADataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["image"]
        if image_path:
            image_path = os.path.join(IMAGE_FOLDER, image_path)
            image = Image.open(image_path).convert("RGB")
        else:
            image = None

        conversations = item["conversations"]
        prompt = ""
        for c in conversations:
            prefix = "USER: " if c["from"] == "human" else "ASSISTANT: "
            prompt += prefix + c["value"] + "\n"

        batch = self.processor(images=image, text=prompt, return_tensors="pt")
        return batch


# ====== Load Model ======
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # ===== LoRA =====
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, processor


# ====== Training ======
def train():
    model, tokenizer, processor = load_model()

    dataset = LLaVADataset(DATA_PATH, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch, labels=batch["input_ids"])
            loss = out.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("loss:", loss.item())

    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()