#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=7

num_layers=28 # Example number of layers for llava-1.5-7b-hf
step=5
num_train=200
num_test=100
model_name="Qwen2.5-VL-7B-Instruct"
inter_start_layer=15
inter_end_layer=27
alpha_text=2.2
dataset="ScienceQA"

# echo "🚀 Step 1: Extracting activations..."
python -m experiments.get_activations_inst --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset

# echo "🧠 Step 2: Training steering vector model..."
# # python -m experiments.train_steering_vector --model_name $model_name --start_layer 0 --end_layer 10 &
# # python -m experiments.train_steering_vector --model_name $model_name --start_layer 10 --end_layer 20 &
# python -m experiments.train_steering_vector --model_name $model_name --start_layer 20 --end_layer 33



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

# python -m experiments.inference_activations --start_layer 0 --end_layer $num_layers --model_name $model_name

echo "✅ All layer groups finished!"

echo "🎯 Step 3: Generating responses with steering vectors applied..."
python -m experiments.generation --model_name $model_name --num_test $num_test --num_train $num_train --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --max_layer $num_layers

echo "🎉 All steps completed!"

done