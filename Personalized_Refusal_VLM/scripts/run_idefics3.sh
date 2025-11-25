#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=7

num_layers=33 # Example number of layers for idefics2-8b
step=10
num_train=200
num_test=100
model_name="Idefics3-8B-Llama3"
inter_start_layer=15
inter_end_layer=32
alpha_text=1.
dataset="ScienceQA"

echo "🚀 Step 1: Extracting activations..."
# python -m experiments.get_activations_inst --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset

python -m experiments.generation --model_name $model_name --num_test $num_test --num_train $num_train --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset