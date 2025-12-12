echo "Running our method..."
bash scripts/ours.sh

echo "Running baseline method..."
bash scripts/persona_baseline.sh

echo "Running refusal score..."
bash scripts/refusal_score.sh