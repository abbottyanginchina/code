echo "Running llava-1.5-7b-hf..."
bash scripts/run_refusal_score_llava_7b.sh

echo "Running llava-1.5-13b-hf..."
bash scripts/run_refusal_score_llava_13b.sh

echo "Running Idefics3-8B-Llama3..."
bash scripts/run_refusal_score_idefics3.sh 