from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("/gpu02home/jmy5701/gpu/models/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("/gpu02home/jmy5701/gpu/models/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "/gpu02home/jmy5701/gpu/data/or-bench/audio/4.mp3"},
        {"type": "text", "text": "Can you repeat the question?"},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audio_path = ele["audio_url"]
                audio, _ = librosa.load(
                    audio_path,
                    sr=processor.feature_extractor.sampling_rate
                )
                audios.append(audio)

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to("cuda")

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("================================================")
print(response)
