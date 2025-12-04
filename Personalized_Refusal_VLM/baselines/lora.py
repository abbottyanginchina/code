import os
import json
import torch
import mmengine
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from vti_utils.utils import get_all_datasets

class LLaVADataset(Dataset):
    def __init__(self, json_path, processor, max_length=2048): # <-- 新增 max_length 参数
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
def train(cfg):
    # -----------------------------
    # 配置
    # -----------------------------
    model_name = f"/gpuhome/jmy5701/gpu/models/{cfg.model_name}"
    json_path = f"/gpuhome/jmy5701/gpu/data/{cfg.data.dataset_name}_{cfg.data.subject}_lora/test_answer.json"           # ← 你的文件
    output_dir = f"../{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}_lora_output"

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

def inference(cfg):
    # -----------------------------
    # 配置
    # -----------------------------
    model_name = f"/gpuhome/jmy5701/gpu/models/{cfg.model_name}"
    lora_dir = f"../{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}_lora_output" # LoRA 权重保存的路径
    
    # -----------------------------
    # Processor & Model
    # -----------------------------
    print("Loading processor and base model...")
    processor = LlavaProcessor.from_pretrained(model_name)
    
    # 加载基础模型
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto", # 自动分配到 GPU
    )

    # -----------------------------
    # 加载 LoRA 权重
    # -----------------------------
    if not os.path.exists(lora_dir):
        print(f"❌ 错误: LoRA 权重目录 '{lora_dir}' 不存在。请先运行 train() 方法。")
        return

    print(f"Loading LoRA weights from: {lora_dir}")
    # 这里的 LoRA 配置不需要像训练时那么完整，但需要确保 PeftModel 能够识别
    peft_config = LoraConfig.from_pretrained(lora_dir) 
    
    # 将 LoRA 权重合并到基础模型中 (作为 PeftModel)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    
    # 建议切换到评估模式
    model.eval()

    # -----------------------------
    # 准备推理输入
    # -----------------------------
    original_data = get_all_datasets(cfg)
    in_test_text = original_data["in_test_text"]
    out_test_text = original_data["out_test_text"]
    in_test_images = original_data["in_test_images"]
    out_test_images = original_data["out_test_images"]

    answers_file = f"../baseline_results/{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/out_of_constraint_answer.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i in range(len(out_test_text)):
        inputs = processor(
            images=out_test_images[i], 
            text=f"<image>\n{out_test_text[i]}\nASSISTANT:", 
            return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,          # 限制生成最大的长度
                do_sample=False,             # 关闭采样，进行确定性解码
                # temperature=0.7,           # 如果启用采样，可以设置温度
            )
        generated_only_ids = output_ids[0][input_len:]
        output_text = processor.decode(generated_only_ids, skip_special_tokens=True)
        
        result = {
            "response": output_text,
            "question": out_test_text[i],
        }
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
        print(output_text)

    ans_file.close()
    print("Inference on in-test data:-----------")

    answers_file = f"../baseline_results/{cfg.model_name}_{cfg.data.dataset_name}_{cfg.data.subject}/in_constraint_answer.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i in range(len(in_test_text)):
        inputs = processor(
            images=in_test_images[i], 
            text=f"<image>\n{in_test_text[i]}\nASSISTANT:", 
            return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,          # 限制生成最大的长度
                do_sample=False,             # 关闭采样，进行确定性解码
                # temperature=0.7,           # 如果启用采样，可以设置温度
            )
        generated_only_ids = output_ids[0][input_len:]
        output_text = processor.decode(generated_only_ids, skip_special_tokens=True)
        result = {
            "response": output_text,
            "question": out_test_text[i],
        }
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
        print(output_text)

    ans_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Lora baseline...")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/gpuhome/jmy5701/gpu/models",
        help="Path to the pretrained models",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=200,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=200,
        help="Number of training samples to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/gpuhome/jmy5701/gpu/data",
        help="Path to the pretrained models",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="biology",
        help="Subject to use",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_path = f'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.num_test is not None:
        cfg.num_test = args.num_test
    if args.num_train is not None:  
        cfg.num_train = args.num_train
    if args.dataset is not None:
        cfg.data.dataset_name = args.dataset
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.subject is not None:
        cfg.data.subject = args.subject


    # train(cfg)

    inference(cfg)

