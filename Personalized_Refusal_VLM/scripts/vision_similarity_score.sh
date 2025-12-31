#!/bin/bash
set -e

num_layers=33 
model_name="llava-1.5-7b-hf"
dataset="ScienceQA"
subject="biology"


python -m experiments.vision_experiments.similarity_score --start_layer 0 --end_layer $num_layers --subject $subject --dataset $dataset --model_name $model_name
