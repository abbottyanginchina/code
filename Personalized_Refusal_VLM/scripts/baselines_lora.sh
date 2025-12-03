#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5

num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
inter_start_layer=12
inter_end_layer=32
alpha_text=1.3
dataset="ScienceQA"
subject="biology"

python -m baselines.data_preperation


python -m baselines.lora --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject