import torch
import soundfile as sf
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "../../../models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# single inference
wavs, sr = model.generate_voice_design(
    text="哥哥，哥哥！快来操我！妹妹想要",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
sf.write("output_voice_design.wav", wavs[0], sr)

audio = AudioSegment.from_wav("output_voice_design.wav")
audio.export("output_voice_design.mp3", format="mp3", bitrate="192k")
