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

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ======================================================
# Dataset（适配你的“human/gpt”格式）
# ======================================================
class LLaVADataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ---- image ----
        image = Image.open(item["image"]).convert("RGB")

        # ---- conversation ----
        conv = item["conversations"]
        user_msg = conv[0]["value"]     # "from": "human"
        assistant_msg = conv[1]["value"]  # "from": "gpt"

        # prompt (已经包含 <image>)
        prompt = user_msg

        # ---- encode input ----
        inputs = self.processor(
            images=image,
            text=prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ---- encode labels ----
        with self.processor.as_target_processor():
            labels = self.processor(
                text=assistant_msg,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        inputs["labels"] = labels

        # Trainer 要求每个 key 是 1D，而不是 [1, seq]
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ======================================================
# 主函数
# ======================================================
def main():
    # -----------------------------
    # 配置
    # -----------------------------
    model_name = "/gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf"
    json_path = "/gpuhome/jmy5701/gpu/data/ScienceQA_biology_lora/test_answer.json"           # ← 你的文件
    output_dir = "./llava_lora_output"

    # -----------------------------
    # Processor & Model
    # -----------------------------
    processor = LlavaProcessor.from_pretrained(model_name)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # -----------------------------
    # LoRA 配置
    # -----------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = LLaVADataset(json_path, processor)

    # -----------------------------
    # Trainer 参数
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,   # ← 必须，否则输入会被 Trainer 过滤掉
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # -----------------------------
    # 保存 LoRA
    # -----------------------------
    trainer.save_model(output_dir)
    print("LoRA saved to:", output_dir)


if __name__ == "__main__":
    main()