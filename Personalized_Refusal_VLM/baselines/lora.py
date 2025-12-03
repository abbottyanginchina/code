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
    def __init__(self, json_path, processor, max_length=2048): # <-- 新增 max_length 参数
        self.data = json.load(open(json_path, "r"))
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length # <-- 保存 max_length
    
    def __len__(self):
        # 确保返回正确的长度
        return len(self.data)

    # def __getitem__(self, idx):
    #     item = self.data[idx]

    #     image = Image.open(item["image"]).convert("RGB")

    #     conv = item["conversations"]
    #     user_msg = conv[0]["value"]            # e.g. "<image>\nDescribe…"
    #     assistant_msg = conv[1]["value"]

    #     # -------------------------------
    #     # 1）processor 自动生成 input_ids (使用统一的 max_length)
    #     # -------------------------------
    #     # 注意：对于 Llava 模型，文本是 user_msg + assistant_msg
    #     # 但是这里您只用 user_msg 传入 processor，这是 Llava 官方微调的常见做法。
    #     # 这里的 max_length 确保了 inputs['input_ids'] 的长度固定。
    #     inputs = self.processor(
    #         images=image,
    #         text=user_msg,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,  # <--- **关键改动 1**
    #     )

    #     # -------------------------------
    #     # 2）用 tokenizer 生成 label (使用统一的 max_length)
    #     # -------------------------------
    #     # 这里的 labels 必须和 input_ids 的长度一致！
    #     labels = self.tokenizer(
    #         assistant_msg,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,  # <--- **关键改动 2**
    #     )["input_ids"]

    #     # 确保 label 序列长度一致后再处理 -100
    #     # 如果模型有图像 token 占位，您可能还需要对 labels 进行进一步处理，
    #     # 使得 labels 的有效部分对齐到 input_ids 的有效部分。
    #     # 但是，最主要的问题是长度不一致，先解决长度。

    #     # 把 pad 位置设为 -100
    #     labels[labels == self.tokenizer.pad_token_id] = -100

    #     inputs["labels"] = labels

    #     return {k: v.squeeze(0) for k, v in inputs.items()}
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        conv = item["conversations"]
        user_msg = conv[0]["value"]            # e.g. "<image>\nDescribe…"
        assistant_msg = conv[1]["value"]

        # ------------------------------------------------
        # 步骤 1：调用 processor 获取图像输入（pixel_values）
        # ------------------------------------------------
        # 即使我们不使用它的 input_ids/attention_mask，也需要调用它来获取图像数据
        inputs = self.processor(
            images=image,
            text=user_msg,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length", 
            truncation=True,
        )

        # ------------------------------------------------
        # 步骤 2：构造 Prompt 的有效长度（用于掩码）
        # ------------------------------------------------
        prompt_inputs = self.processor(
            images=image,
            text=user_msg,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            # 不用 padding，方便计算有效长度
        )
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        # 计算 Prompt 的有效长度（不含 PAD token）
        prompt_len = (prompt_ids != self.tokenizer.pad_token_id).sum().item()
        prompt_ids_valid = prompt_ids[:prompt_len]

        # ------------------------------------------------
        # 步骤 3：构造 Response 和完整的序列
        # ------------------------------------------------
        response_inputs = self.tokenizer(
            assistant_msg, 
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        # 拼接最终的 input_ids
        input_ids = torch.cat([
            prompt_ids_valid, 
            response_inputs, 
            torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        ])

        # ------------------------------------------------
        # 步骤 4：构造 Labels 和进行掩码
        # ------------------------------------------------
        labels = torch.cat([
            torch.full((prompt_len,), -100, dtype=torch.long), # Prompt Masking
            response_inputs,
            torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        ])
        
        # ------------------------------------------------
        # 步骤 5：统一长度（Padding/Truncation）
        # ------------------------------------------------
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        padding_len = self.max_length - input_ids.shape[0]
        if padding_len > 0:
            input_ids = torch.cat([
                input_ids, 
                torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            labels = torch.cat([
                labels, 
                torch.full((padding_len,), -100, dtype=torch.long) # PAD Labels 也要设为 -100
            ])
        
        # ------------------------------------------------
        # 步骤 6：更新 inputs 字典（修复 NameError 的关键）
        # ------------------------------------------------
        
        # 将我们手动构造的序列替换掉 processor 自动生成的序列
        inputs["input_ids"] = input_ids.unsqueeze(0)
        
        # 构造 Attention Mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).int().unsqueeze(0)
        inputs["attention_mask"] = attention_mask
        
        # 添加 labels
        inputs["labels"] = labels.unsqueeze(0)
        
        # 确保返回字典中的所有值都是单批次的 Tensor
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