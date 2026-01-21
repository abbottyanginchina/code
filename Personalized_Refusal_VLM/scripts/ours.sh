# export CUDA_VISIBLE_DEVICES=6

echo "Running Idefics3-8B-Llama3..."
bash scripts/run_idefics3.sh

echo "Running llava-1.5-13b-hf..."
bash scripts/run_llava_1.5_13b_hf.sh

echo "Running InstructBLIP..."
bash scripts/run_instructBLIP.sh

echo "Running llava-1.5-7b-hf..."
bash scripts/run_llava_1.5_7b_hf.sh