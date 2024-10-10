#!/bin/bash

MAX_ITER=2000
C=1
SEED_list=(1 2 3 4 5 42 27 128 256 512) #
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
    "Qwen/Qwen-VL-Chat"
    "Qwen/Qwen-7B-Chat"
    "Qwen/Qwen1.5-7B-Chat"
    "openbmb/MiniCPM-Llama3-V-2_5"
)

for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=3,4 python probe.py --model_name "$model_name" --max_iter $MAX_ITER --C $C --seed $SEED 
        CUDA_VISIBLE_DEVICES=3,4 python probe.py --model_name "$model_name" --max_iter $MAX_ITER --C $C --seed $SEED --abstract_only 
        CUDA_VISIBLE_DEVICES=3,4 python probe.py --model_name "$model_name" --max_iter $MAX_ITER --C $C --seed $SEED --concrete_only 
    done
done