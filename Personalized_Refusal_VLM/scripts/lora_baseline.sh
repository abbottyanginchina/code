export CUDA_VISIBLE_DEVICES=5

echo "🎉 Baseline Lora started!"

# echo "LLAVA 7B"
# bash ./scripts/lora_llava_7b.sh

echo "LLAVA 13B"
bash ./scripts/lora_llava_13b.sh

echo "Idefics3"
bash ./scripts/lora_idefics3.sh

echo "🎉 Baseline Lora completed!"