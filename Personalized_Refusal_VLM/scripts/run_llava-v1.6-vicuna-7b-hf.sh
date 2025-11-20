#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=7

num_layers=37 # Example number of layers for llava-1.5-7b-hf
step=13
num_train=200
num_test=100
model_name="llava-v1.6-vicuna-7b-hf"
inter_start_layer=12
inter_end_layer=35
alpha_text=1.8
dataset="ScienceQA"

echo "🚀 Step 1: Extracting activations..."
python -m experiments.get_activations --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset

echo "🧠 Step 2: Training steering vector model..."
# python -m experiments.train_steering_vector --model_name $model_name --start_layer 36 --end_layer 37
# python -m experiments.train_steering_vector --model_name $model_name --start_layer 18 --end_layer 20 &
# python -m experiments.train_steering_vector --model_name $model_name --start_layer 28 --end_layer 30 &
# wait


# for ((layer=0; layer<num_layers; layer+=step)); do
#     end_layer=$((layer + step))

#     if [ $end_layer -gt $num_layers ]; then
#         end_layer=$num_layers
#     fi

#     echo "🔄 Training layers $layer → $end_layer ..."
    
#     python -m experiments.train_steering_vector \
#         --model_name $model_name \
#         --start_layer $layer \
#         --end_layer $end_layer & # ← 并行运行
# done
# wait

echo "Step 3: Inference activations with steering vectors applied..."
# python -m experiments.inference_activations --start_layer 0 --end_layer $num_layers --model_name $model_name

echo "✅ All layer groups finished!"

echo "🎯 Step 4: Generating responses with steering vectors applied..."
python -m experiments.generation --model_name $model_name --num_test $num_test --num_train $num_train --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --max_layer $num_layers

echo "🎉 All steps completed!"

done