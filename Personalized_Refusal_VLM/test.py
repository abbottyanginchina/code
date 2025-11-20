
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    "/gpuhome/jmy5701/gpu/models/Qwen-VL-Chat",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "/gpuhome/jmy5701/gpu/models/Qwen-VL-Chat",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# -------------------------
# 1. 先构造 prompt（string）
# -------------------------
prompt = tokenizer.from_list_format([
    {"image": 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {"text": "这是什么？"}
])

# -------------------------
# 2. 再 tokenize（才能得到 input_ids + pixel_values）
# -------------------------
enc = tokenizer(
    prompt,
    return_tensors="pt"
)
# import pdb; pdb.set_trace()

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
# pixel_values = enc["pixel_values"].to(device)
image = Image.open("demo.jpeg").convert("RGB")
pixel_values = model.transformer.visual.process_images([image]).to(device)

# -------------------------
# 3. 构造拒绝 labels（Teacher Forcing）
# -------------------------
refusal = "对不起，我不能回答这个问题。"
refusal_ids = tokenizer(
    refusal,
    return_tensors="pt",
    add_special_tokens=False
).input_ids.to(device)

labels = torch.full_like(input_ids, -100)
labels[:, -refusal_ids.size(1):] = refusal_ids   # 对齐末尾 token

# -------------------------
# 4. forward + 获取隐藏层
# -------------------------
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        output_hidden_states=True,
        return_dict=True
    )

hidden_states = outputs.hidden_states
print(f"Total layers = {len(hidden_states)}")
print(hidden_states[-1].shape)  # [1, seq_len, hidden_dim]