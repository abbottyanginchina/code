export CUDA_VISIBLE_DEVICES=5

# echo "🚀 Step 1: Generating responses for llava-1.5-7b-hf..."
# bash scripts/sys_llava_7b.sh

echo "🚀 Step 2: Generating responses for llava-1.5-13b-hf..."
bash scripts/sys_llava_13b.sh

echo "🚀 Step 3: Generating responses for Idefics3-8B-Llama3..."
bash scripts/sys_idefics3.sh