from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

# ------------------------------------------------------------------
# 1. 构造图像 + query
# ------------------------------------------------------------------
query = tokenizer.from_list_format([
    {"image": "demo.png"},   # 你的训练图片路径
    {"text": "这是什么？"}
])

input_ids = query["input_ids"].to(device)
pixel_values = query["pixel_values"].to(device)
attention_mask = query["attention_mask"].to(device)

# ------------------------------------------------------------------
# 2. 构造 teacher forcing 的拒绝句子
# ------------------------------------------------------------------
refusal_text = "对不起，我无法回答这个问题。"  # 你的拒绝模板
refusal_ids = tokenizer(
    refusal_text,
    return_tensors="pt",
    add_special_tokens=False
).input_ids.to(device)

# 把 label 对齐到 input_ids 的长度
labels = torch.full_like(input_ids, -100)   # -100 表示忽略
labels[:, -refusal_ids.size(1):] = refusal_ids  # 让最后 K 个 token 预测拒绝句

# ------------------------------------------------------------------
# 3. 调用 forward 做 teacher forcing（激活隐藏层）
# ------------------------------------------------------------------
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states   # tuple: [layer0, layer1, ...]