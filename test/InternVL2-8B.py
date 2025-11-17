import torch
from transformers import AutoTokenizer, AutoModel
path = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
# 假设你有输入文本
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
inputs = tokenizer("示例文本", return_tensors="pt").to("cuda")

# 获取hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # 是一个tuple，每层的hidden state

# 例如，最后一层的hidden state
last_hidden_state = hidden_states[-1]
print(last_hidden_state.shape)