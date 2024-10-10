#!/bin/bash

model_list=(
    "gpt-3.5-turbo"
    "gpt-4-turbo"
    "gpt-4o"
    "gpt-4o-mini"

)

for model_name in "${model_list[@]}"; do
    echo "Running model: $model_name"
 
    for i in { 1..10}
    do
        output_dir="output_prompt/abstract2concrete/$i"
        CUDA_VISIBLE_DEVICES=4 python test_chatgpt_abstract2concrete_api.py --model_name $model_name  --output_dir "$output_dir"

    done

 
    for i in { 1..10}
    do
        output_dir="output_prompt/abstract2concrete_prompt_size/$i"
        CUDA_VISIBLE_DEVICES=4 python test_chatgpt_abstract2concrete_api.py --model_name $model_name --output_dir "$output_dir" --if_prompt_size
        
        
    done

 
    for i in { 1..10}
    do
        output_dir="output_prompt/abstract2concrete_extend/$i"
        CUDA_VISIBLE_DEVICES=4 python test_chatgpt_abstract2concrete_api.py --model_name $model_name --output_dir "$output_dir" --if_extend

    done


 
    for i in { 1..10}
    do
        output_dir="output_prompt/abstract2concrete_extend_prompt_size/$i"
        CUDA_VISIBLE_DEVICES=4 python test_chatgpt_abstract2concrete_api.py --model_name $model_name --output_dir "$output_dir" --if_prompt_size --if_extend
    done

done
