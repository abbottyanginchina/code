set -e
export CUDA_VISIBLE_DEVICES=6

model_name="llava-1.5-7b-hf"
num_test=200
num_train=200
inter_start_layer=12
inter_end_layer=32
alpha_text=1.7
dataset="ScienceQA"
subject="biology"

process_subject() {
    dataset=$1
    subject=$2
    model_name=$3
    
    python -m experiments.generate_refusal_score --model_name $model_name --num_test $num_test --num_train $num_train \
        --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --subject $subject
}

dataset="ScienceQA"
subjects=("biology" "geography" "physics")  
model_name=("llava-1.5-7b-hf" "llava-v1.6-vicuna-7b-hf" "Idefics3-8B-Llama3")
for subject in "${subjects[@]}"; do
    for model_name in "${model_names[@]}"; do
        process_subject "$dataset" "$subject" "$model_name"
    done
done

dataset="MMMU"
subjects=("Math" "Geography" "Art_Theory") 
model_name=("llava-1.5-7b-hf" "llava-v1.6-vicuna-7b-hf" "Idefics3-8B-Llama3")
for subject in "${subjects[@]}"; do
    for model_name in "${model_names[@]}"; do
        process_subject "$dataset" "$subject" "$model_name"
    done
done
echo "✅ All finished!"