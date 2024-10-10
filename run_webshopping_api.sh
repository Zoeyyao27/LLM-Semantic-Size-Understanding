################
output_dir="output_webshopping/qwen_api"
##api model
SEED_list=(1 0 2 3 4)
model_list=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-3.5-turbo"
)
for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED --if_text --if_no_figure
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED --if_text 
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED
    done
done


for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED --if_text --if_no_figure
    done
done


for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED --if_text 
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --output_path $output_dir --model_path "$model_name" --random_seed $SEED
    done
done


