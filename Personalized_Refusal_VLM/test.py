from datasets import load_dataset, DatasetDict

data_files = {
    "train": "/gpu02home/jmy5701/gpu/data/ScienceQA/data/train-*.parquet",
    "validation": "/gpu02home/jmy5701/gpu/data/ScienceQA/data/validation-*.parquet",
    "test": "/gpu02home/jmy5701/gpu/data/ScienceQA/data/test-*.parquet",
}

ds = load_dataset("parquet", data_files=data_files)

# 保存成 HuggingFace 原生格式
ds.save_to_disk("/gpu02home/jmy5701/gpu/data/ScienceQA_hf")