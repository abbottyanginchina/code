from datasets import load_dataset

dataset = load_dataset(f"/gpu02home/jmy5701/gpu/data/ScienceQA")["train"].filter(lambda e: e["image"] is not None)