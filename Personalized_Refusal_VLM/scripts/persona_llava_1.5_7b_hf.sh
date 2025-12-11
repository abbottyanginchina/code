#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=6

num_layers=33 # Example number of layers for llava-1.5-7b-hf
step=11
num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
inter_start_layer=12
inter_end_layer=32
alpha_text=1.3

process_subject() {
    dataset=$1
    subject=$2
    
    echo "🎉 Dataset $dataset, subject $subject started!"
    echo "🚀 Step 1: Extracting activations..."
    python -m experiments.get_activations --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject


    echo "🎯 Step 2: Generating responses with steering vectors applied..."
    python -m baselines.Persona_generation --model_name $model_name --num_test $num_test --num_train $num_train \
        --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --subject $subject

    echo "🎉 Dataset $dataset, subject $subject completed!"
    echo ""
}


dataset="ScienceQA"
subjects=("biology" "geography" "physics")  
# subjects=("biology")  
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

echo "🎉 All steps completed!"



