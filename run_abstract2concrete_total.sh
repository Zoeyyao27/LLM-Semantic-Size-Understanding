#!/bin/bash
model_list=(
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "lmsys/vicuna-7b-v1.5"
    "llava-hf/llava-v1.6-vicuna-7b-hf"
    "01-ai/Yi-6B-Chat"
    "model/Yi-VL-6B"
    "lmsys/vicuna-13b-v1.5"
    "llava-hf/llava-v1.6-vicuna-13b-hf"
    "01-ai/Yi-34B-Chat"
    "model/Yi-VL-34B"
    "THUDM/cogvlm2-llama3-chat-19B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen-7B-Chat"
    "Qwen/Qwen1.5-7B-Chat"
    "Qwen/Qwen-VL-Chat"
    "openbmb/MiniCPM-Llama3-V-2_5"
)

OUTPUT_DIR="output_prompt_qwen2"


################original dataset#################
for model_name in "${model_list[@]}"; do
    echo "Running model: $model_name"
    for i in {1..10}
    do
        output_dir="$OUTPUT_DIR/abstract2concrete_prompt_size/$i"
        
        CUDA_VISIBLE_DEVICES=1 python test_abstract2concrete_size.py --model_name "$model_name" \
        --output_dir "$output_dir" --if_prompt_size

    done
done


for model_name in "${model_list[@]}"; do
    echo "Running model: $model_name"
    for i in {1..10}
    do
        output_dir="$OUTPUT_DIR/abstract2concrete/$i"
        
        CUDA_VISIBLE_DEVICES=1 python test_abstract2concrete_size.py --model_name "$model_name" \
        --output_dir "$output_dir"

    done
done

##################extend dataset###############
for model_name in "${model_list[@]}"; do
    echo "Running model: $model_name"
    for i in {1..10}
    do
        output_dir="$OUTPUT_DIR/abstract2concrete_extend/$i"
        
        CUDA_VISIBLE_DEVICES=1 python test_abstract2concrete_size.py --model_name "$model_name" \
        --output_dir "$output_dir" --if_extend

    done
done

for model_name in "${model_list[@]}"; do
    echo "Running model: $model_name"
    for i in {1..10}
    do
        output_dir="$OUTPUT_DIR/abstract2concrete_extend_prompt_size/$i"
        
        CUDA_VISIBLE_DEVICES=1 python test_abstract2concrete_size.py --model_name "$model_name" \
        --output_dir "$output_dir" --if_prompt_size --if_extend

    done
done




