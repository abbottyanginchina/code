import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

device = "cuda"

# -----------------------------
# 1. 加载模型和 processor
# -----------------------------
processor = AutoProcessor.from_pretrained("/gpuhome/jmy5701/gpu/models/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "/gpuhome/jmy5701/gpu/models/idefics2-8b",
    output_hidden_states=True,       # <<--- 必须开这个！
    torch_dtype=torch.float16
).to(device).eval()

# -----------------------------
# 2. 准备输入图片
# -----------------------------
image1 = Image.open("/gpuhome/jmy5701/img1.jpg")

# -----------------------------
# 3. 构建 prompt（与 generate 一致）
# -----------------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
]

# 最终我们要 teacher forcing 的“拒绝标签”
refusal_text = "I cannot answer."

# -----------------------------
# 4. 将文本 + 图像 转成模型输入
# -----------------------------
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

enc = processor(
    text=prompt,
    images=[image1],
    return_tensors="pt",
    padding=True
).to(device)

input_ids = enc["input_ids"]

# -----------------------------
# 5. 构造 teacher forcing labels
# -----------------------------
refusal_ids = processor.tokenizer(
    refusal_text,
    add_special_tokens=False,
    return_tensors="pt"
)["input_ids"].to(device)

labels = input_ids.clone()
# 仅强制最后 refusal_ids 的 token
labels[:, -refusal_ids.size(1):] = refusal_ids

# -----------------------------
# 6. 前向传播（得到 hidden states）
# -----------------------------
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        pixel_values=enc["pixel_values"],
        attention_mask=enc["attention_mask"],
        labels=labels,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states  # tuple: [num_layers][batch, seq_len, hidden]

print("Total layers:", len(hidden_states))
print("Shape of last layer hidden states:", hidden_states[-1].shape)