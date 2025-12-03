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
    def __init__(self, json_path, processor):
        self.data = json.load(open(json_path, "r"))
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")

        conv = item["conversations"]
        user_msg = conv[0]["value"]
        assistant_msg = conv[1]["value"]

        # prompt (<image> 已包含)
        full_prompt = user_msg + "\nASSISTANT: "
        full_answer = assistant_msg

        # -------------------------
        # 1) tokenize prompt
        # -------------------------
        prompt_ids = self.tokenizer(
            full_prompt,
            add_special_tokens=False
        )["input_ids"]

        # -------------------------
        # 2) tokenize answer
        # -------------------------
        answer_ids = self.tokenizer(
            full_answer,
            add_special_tokens=False
        )["input_ids"]

        # -------------------------
        # 3) 合并：input = prompt + answer
        # -------------------------
        input_ids = prompt_ids + answer_ids

        # -------------------------
        # 4) 构造 labels
        #    - prompt 部分 = -100
        #    - answer 部分 = answer_ids
        # -------------------------
        labels = [-100] * len(prompt_ids) + answer_ids

        # pad 到固定长度（随便给一个，比如 2048，不要太小）
        max_len = 2048
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        padding_len = max_len - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len

        # -------------------------
        # image encoding
        # -------------------------
        image_tensor = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "pixel_values": image_tensor,   # LLaVA 必须加这个字段
        }


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