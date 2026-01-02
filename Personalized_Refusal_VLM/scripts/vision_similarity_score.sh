#!/bin/bash
set -e

num_layers=33 
model_name="llava-1.5-7b-hf"

process_subject() {
    dataset=$1
    subject=$2
    python -m experiments.vision_experiments.similarity_score --start_layer 0 --end_layer $num_layers --subject $subject --dataset $dataset --model_name $model_name
}

dataset="ScienceQA"
subjects=("biology" "geography" "physics")  
# subjects=("physics") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

echo "🎉 All steps completed!"