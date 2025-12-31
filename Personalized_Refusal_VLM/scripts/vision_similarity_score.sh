#!/bin/bash
set -e


num_layers=33 # Example number of layers for llava-1.5-7b-hf
step=11
num_train=200
num_test=200
model_name="llava-1.5-7b-hf"
inter_start_layer=12
inter_end_layer=32
alpha_text=1.7


python -m experiments.vision_experiments.similarity_score --start_layer 0 --end_layer $num_layers --subject $subject --dataset $dataset --model_name $model_name
