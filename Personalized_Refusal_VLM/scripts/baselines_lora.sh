#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=6

num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
dataset="ScienceQA"
subject="biology"

process() {
    dataset=$1
    subject=$2
    
    # python -m baselines.data_preperation --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
    # wait
    python -m baselines.lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
    wait
}

dataset="ScienceQA"
subjects=("biology" "geography" "physics")  # 根据需要修改这里的 subjects
for subject in "${subjects[@]}"; do
    process "$dataset" "$subject"
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory")  # 根据需要修改这里的 subjects
for subject in "${subjects[@]}"; do
    process "$dataset" "$subject"
done

echo "🎉 All steps completed!"