import os
import torch
import soundfile as sf
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

root_dir = "/gpu02home/jmy5701/gpu/"
model = Qwen3TTSModel.from_pretrained(
    os.path.join(root_dir, "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# # single inference
# wavs, sr = model.generate_voice_design(
#     text="哥哥，哥哥！快来操我！妹妹想要",
#     language="Chinese",
#     instruct="体现撒娇稚嫩的女朋友的声线，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
# )
# sf.write("output_voice_design.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]
)
sf.write(os.path.join(root_dir, "audio_results", "audio_results/output_voice_design_1.wav"), wavs[0], sr)
sf.write(os.path.join(root_dir, "audio_results", "audio_results/output_voice_design_2.wav"), wavs[1], sr)