#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5,6

num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
dataset="ScienceQA"
subject="biology"

# python -m baselines.data_preperation --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
python -m baselines.lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject

subject="physics"
python -m baselines.data_preperation --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
python -m baselines.lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject

subject="geography"
python -m baselines.data_preperation --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject
python -m baselines.lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject