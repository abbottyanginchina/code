#!/bin/bash
set -e

num_train=200
num_test=200
model_name="Idefics3-8B-Llama3"
# dataset="ScienceQA"
# subject="biology"

process() {
    dataset=$1
    subject=$2
    
    echo "🎉 Dataset $dataset, subject $subject started!"
    # python -m baselines.data_preperation --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
    # wait
    python -m baselines.Idefics3_lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
    wait
    echo "🎉 Dataset $dataset, subject $subject completed!"
}

# dataset="ScienceQA"
# subjects=("biology" "geography" "physics")  # 根据需要修改这里的 subjects
# for subject in "${subjects[@]}"; do
#     process "$dataset" "$subject"
# done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory")  # 根据需要修改这里的 subjects
for subject in "${subjects[@]}"; do
    process "$dataset" "$subject"
done

echo "🎉 All steps completed!"