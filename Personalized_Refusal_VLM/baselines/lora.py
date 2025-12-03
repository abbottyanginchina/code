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

# ===========================
# Load Model & Processor
# ===========================

model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
processor = LlavaProcessor.from_pretrained(model_name)

# ===========================
# LoRA Config
# ===========================
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# ===========================
# Load Dataset
# ===========================
data = load_dataset("json", data_files=train_json)['train']

def preprocess(example):
    conv = example["conversations"]
    img_path = example["image"]

    # Build chat template
    prompt = processor.apply_chat_template(
        conv,
        add_generation_prompt=True
    )
    import pdb; pdb.set_trace()

    # Load image
    image = Image.open(img_path).convert("RGB")
    processed = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    
    return {
        "input_ids": processed["input_ids"][0],
        "pixel_values": processed["pixel_values"][0]
    }

data = data.map(preprocess)

# ===========================
# Training Arguments
# ===========================
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data,
)

# ===========================
# Train
# ===========================
trainer.train()

# ===========================
# Save LoRA Model
# ===========================
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("✅ LoRA training completed.")
print(f"📁 Saved to: {output_dir}")