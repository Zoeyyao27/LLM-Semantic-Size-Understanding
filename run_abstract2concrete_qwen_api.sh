#!/bin/bash


 
for i in {1..10}
do
 
    output_dir="output_prompt/abstract2concrete_extend/$i"

 
    CUDA_VISIBLE_DEVICES=4 python test_VL_plus_abstract2concrete_api.py --model_name qwen-vl-max --output_dir "$output_dir" --if_extend

 
    CUDA_VISIBLE_DEVICES=4 python test_VL_plus_abstract2concrete_api.py --model_name qwen-vl-plus --output_dir "$output_dir" --if_extend


done


for i in {1..10}
do
 
    output_dir="output_prompt/abstract2concrete_extend_prompt_size/$i"

 
    CUDA_VISIBLE_DEVICES=4 python test_VL_plus_abstract2concrete_api.py --model_name qwen-vl-max --output_dir "$output_dir" --if_prompt_size --if_extend
    
    CUDA_VISIBLE_DEVICES=4 python test_VL_plus_abstract2concrete_api.py --model_name qwen-vl-plus --output_dir "$output_dir" --if_prompt_size --if_extend
    
done
