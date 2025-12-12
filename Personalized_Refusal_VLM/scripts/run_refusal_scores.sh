set -e
export CUDA_VISIBLE_DEVICES=6

python -m experiments.generation --model_name $model_name --num_test $num_test --num_train $num_train \
        --inter_start_layer $inter_start_layer --inter_end_layer $inter_end_layer --alpha_text $alpha_text --dataset $dataset --subject $subject