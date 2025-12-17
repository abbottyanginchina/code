#!/bin/bash
set -e

num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
model_path="../../models"
data_path="../../data"

process_subject() {
    dataset=$1
    subject=$2
    
    echo "🎉 Dataset $dataset, subject $subject started!"
    echo "🚀 Step 1: Generating responses..."
    python -m experiments.prompt_generation --model_name $model_name --model_path $model_path --data_path $data_path --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject

    echo "🎉 Dataset $dataset, subject $subject completed!"
    echo ""
}


dataset="ScienceQA"
subjects=("biology" "geography" "physics")  
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

echo "🎉 All steps completed!"



