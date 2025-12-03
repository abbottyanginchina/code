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

class LLaVADataset(Dataset):
    def __init__(self, json_path, processor, max_length=256): # <-- 新增 max_length 参数
        self.data = json.load(open(json_path, "r"))
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length # <-- 保存 max_length
    
    def __len__(self):
        # 确保返回正确的长度
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")

        conv = item["conversations"]
        user_msg = conv[0]["value"]            # e.g. "<image>\nDescribe…"
        assistant_msg = conv[1]["value"]

        # -------------------------------
        # 1）processor 自动生成 input_ids (使用统一的 max_length)
        # -------------------------------
        # 注意：对于 Llava 模型，文本是 user_msg + assistant_msg
        # 但是这里您只用 user_msg 传入 processor，这是 Llava 官方微调的常见做法。
        # 这里的 max_length 确保了 inputs['input_ids'] 的长度固定。
        inputs = self.processor(
            images=image,
            text=user_msg,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # <--- **关键改动 1**
        )

        # -------------------------------
        # 2）用 tokenizer 生成 label (使用统一的 max_length)
        # -------------------------------
        # 这里的 labels 必须和 input_ids 的长度一致！
        labels = self.tokenizer(
            assistant_msg,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # <--- **关键改动 2**
        )["input_ids"]

        # 确保 label 序列长度一致后再处理 -100
        # 如果模型有图像 token 占位，您可能还需要对 labels 进行进一步处理，
        # 使得 labels 的有效部分对齐到 input_ids 的有效部分。
        # 但是，最主要的问题是长度不一致，先解决长度。

        # 把 pad 位置设为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

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
    output_dir = "../../llava_lora_output"

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
        num_train_epochs=3,
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