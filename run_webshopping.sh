

SEED_list=(0 1 2 3 4 42 27 128 256 512)
model_list=(
    "model/Yi-VL-6B"
    "model/Yi-VL-34B"
    "gpt-4o"
    "gpt-4o-mini"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "llava-hf/llava-v1.6-vicuna-7b-hf"
    "llava-hf/llava-v1.6-vicuna-13b-hf"
    "THUDM/cogvlm2-llama3-chat-19B"
    "openbmb/MiniCPM-Llama3-V-2_5"
    "Qwen/Qwen-VL-Chat"

)
for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --model_path "$model_name" --random_seed $SEED --if_text
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --model_path "$model_name" --random_seed $SEED --if_text --if_no_figure
    done
done



SEED_list=(0 1 2 3 4 42 27 128 256 512)
model_list=(
    "model/Yi-VL-6B"
    "model/Yi-VL-34B"
    "gpt-4o"
    "gpt-4o-mini"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "llava-hf/llava-v1.6-vicuna-7b-hf"
    "llava-hf/llava-v1.6-vicuna-13b-hf"
    "gpt-4o-mini"
    "THUDM/cogvlm2-llama3-chat-19B"
    "openbmb/MiniCPM-Llama3-V-2_5"
    "Qwen/Qwen-VL-Chat"
)
for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --model_path "$model_name" --random_seed $SEED
    done
done


# ##text only
SEED_list=(0 1 2 3 4 42 27 128 256 512)
model_list=(
    "mistralai/Mistral-7B-Instruct-v0.2"
    "lmsys/vicuna-7b-v1.5"
    "lmsys/vicuna-13b-v1.5"
    "01-ai/Yi-6B-Chat"
    "01-ai/Yi-34B-Chat"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen-7B-Chat"
    "Qwen/Qwen1.5-7B-Chat"
)
for SEED in "${SEED_list[@]}"; do
    echo "Running seed: $SEED"
    for model_name in "${model_list[@]}"; do
        echo "Running model: $model_name"
        CUDA_VISIBLE_DEVICES=1 python web_shopping/test_webshopping_vlm.py --model_path "$model_name" --random_seed $SEED --if_text --if_no_figure
    done
done
