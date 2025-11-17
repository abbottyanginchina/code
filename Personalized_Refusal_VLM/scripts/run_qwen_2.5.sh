#!/bin/bash
set -e

num_layers=15 # Example number of layers for llava-1.5-7b-hf
step=3
num_train=200
num_test=100
model_name="Qwen2.5-VL-7B-Instruct"
inter_start_layer=20
inter_end_layer=28
alpha_text=2.0

# echo "🚀 Step 1: Extracting activations..."
python -m experiments.get_activations --model_name $model_name --num_test $num_test --num_train $num_train

# echo "🧠 Step 2: Training steering vector model..."
# # python -m experiments.train_steering_vector --model_name $model_name --start_layer 0 --end_layer 10 &
# # python -m experiments.train_steering_vector --model_name $model_name --start_layer 10 --end_layer 20 &
# python -m experiments.train_steering_vector --model_name $model_name --start_layer 29 --end_layer 30



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


# echo "✅ All layer groups finished!"

echo "🎯 Step 3: Generating responses with steering vectors applied..."
python -m experiments.generation --model_name $model_name --num_test $num_test --num_train $num_train --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text

echo "🎉 All steps completed!"

done