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

process_subject() {
    dataset=$1
    subject=$2
    
    echo "🎉 Dataset $dataset, subject $subject started!"
    echo "🚀 Step 1: Extracting activations..."
    python -m experiments.orth_experiments.orth_get_activations_inst --model_name $model_name --num_test $num_test --num_train $num_train --dataset $dataset --subject $subject

    # echo "🧠 Step 2: Training steering vector model..."
    # python -m experiments.train_steering_vector --model_name $model_name --start_layer 12 --end_layer 19 &
    # python -m experiments.train_steering_vector --model_name $model_name --start_layer 19 --end_layer 26 &
    # python -m experiments.train_steering_vector --model_name $model_name --start_layer 26 --end_layer 33 &
    # wait

    for ((layer=0; layer<num_layers; layer+=step)); do
        end_layer=$((layer + step))

        if [ $end_layer -gt $num_layers ]; then
            end_layer=$num_layers
        fi

        echo "🔄 Training layers $layer → $end_layer ..."
        
        python -m experiments.orth_experiments.orth_train_steering_vector \
            --model_name $model_name \
            --subject $subject \
            --dataset $dataset \
            --start_layer $layer \
            --end_layer $end_layer & # ← 并行运行
    done
    wait

    echo "✅ Step 3: Inference activations with steering vectors applied..."
    python -m experiments.orth_experiments.orth_inference_activations --start_layer 0 --end_layer $num_layers --subject $subject --dataset $dataset --model_name $model_name
    echo "✅ All layer groups finished!"

    echo "🎯 Step 4: Generating responses with steering vectors applied..."
    python -m experiments.orth_experiments.orth_generation --model_name $model_name --num_test $num_test --num_train $num_train \
        --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --subject $subject

    echo "🎉 Dataset $dataset, subject $subject completed!"
    echo ""
}


dataset="ScienceQA"
subjects=("biology" "geography" "physics")  
# subjects=("physics") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory") 
for subject in "${subjects[@]}"; do
    process_subject "$dataset" "$subject"
done

echo "🎉 All steps completed!"



