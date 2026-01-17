import os
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")


def qwen_audio(audio_path):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        # BytesIO(urlopen(ele['audio_url']).read()), 
                        ele['audio_url'],
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(response)

if __name__ == "__main__":
    output_dir = "../../audio_results"
    answers_file = os.path.join(output_dir, f"responses.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for idx in range(200):
        audio_path = os.path.join("../../data/or-bench/audio", f"{idx}.mp3")
        response = qwen_audio(audio_path)
        