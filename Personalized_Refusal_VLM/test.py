from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载分词器和模型
model_name = "/gpuhome/jmy5701/gpu/models/Qwen-VL-Chat"  # 或者 "Qwen/Qwen-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True, device_map="cuda").eval()

# 设置模型为评估模式
model.eval()

# 准备输入文本
input_text = "What is your response to the question: 'What is the capital of France?'"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 获取输入的 token IDs
input_ids = inputs['input_ids']

# 使用 teacher forcing 模式，将 input_ids 作为 labels
# 注意：在 teacher forcing 中，我们通常将目标 token 作为输入，并预测下一个 token
labels = tokenizer("I cannot answer", return_tensors="pt")['input_ids']

# 前向传播，获取 hidden states
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)

# 获取所有层的 hidden states
hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)

# 打印每一层的 hidden state 形状
for i, hidden_state in enumerate(hidden_states):
    print(f"Layer {i} Hidden State Shape: {hidden_state.shape}")