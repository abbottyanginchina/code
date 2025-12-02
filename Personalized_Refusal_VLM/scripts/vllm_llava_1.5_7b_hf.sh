export CUDA_VISIBLE_DEVICES=5

python -m vllm.entrypoints.api_server \
    --model /gpuhome/jmy5701/gpu/models/llava-1.5-7b-hf \
    --served-model-name llava-1.5-7b-hf \
    --dtype float16 \
    --swap-space 32 \
    --dtype float16 \