import os
from datasets import load_dataset
from tqdm import tqdm
import random
from omegaconf import OmegaConf

from utils.extract_words import get_highlighted_text
from utils.audio_generation import generation_audio 
from backends.audio_generation import generate_qwen3_tts_audio

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/cfgs.yaml")

    # text = "Hello, I am Xiaomi, nice to meet you."
    data_path = "../../data/or-bench"
    dataset = load_dataset(data_path, 'or-bench-hard-1k')['train']

    if not os.path.exists(os.path.join(data_path, 'audio')):
        os.makedirs(os.path.join(data_path, 'audio'))

    # Fix random seed
    random.seed(cfg.seed)
    # Sample 200 samples
    num_samples = 200
    dataset = dataset.shuffle(seed=cfg.seed).select(range(num_samples))

    for idx in tqdm(range(len(dataset)), desc="Generating audio", total=len(dataset)):
        import pdb; pdb.set_trace()
        prompt = dataset[idx]['prompt']

        save_path = os.path.join(data_path, 'audio', f'{idx}.mp3')
        # get highlighted text
        # highlighted_text = get_highlighted_text(prompt)
        # print("highlighted_text: ", highlighted_text)
        # Generate audio
        generation_audio(prompt, save_path)