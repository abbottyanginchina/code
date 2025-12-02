export CUDA_VISIBLE_DEVICES=5

python -m vllm.entrypoints.api_server \
    --model /gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf \
    --vision-language