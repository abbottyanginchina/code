import torch
from transformers import AutoTokenizer, AutoModel

path = "/gpuhome/jmy5701/gpu/models/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

# 用于保存 activation
activations = {}

def hook_fn(module, input, output):
    activations['hidden_state'] = output

# 以 Bert 为例，假设你要 hook encoder 的第 0 层
# 具体层名请根据你的模型结构调整
layer = model.encoder.layer[0]
handle = layer.register_forward_hook(hook_fn)

# 输入示例
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
inputs = tokenizer("Hello world!", return_tensors="pt").to('cuda')

with torch.no_grad():
    outputs = model(**inputs)

# 现在 activations['hidden_state'] 就是你要的 hidden state
print(activations['hidden_state'].shape)

# 记得移除 hook
handle.remove()
