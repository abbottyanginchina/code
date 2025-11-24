#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=7

num_layers=32 # Example number of layers for idefics2-8b
step=10
num_train=200
num_test=100
model_name="idefics2-8b"
inter_start_layer=12
inter_end_layer=31
alpha_text=2.5
dataset="ScienceQA"

echo "🚀 Step 1: Extracting activations..."
python -m experiments.get_activations_inst --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset