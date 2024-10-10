import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir="cache",torch_dtype=torch.float16,
                                                 device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto",cache_dir="cache")
    return tokenizer, model

                                                 

def single_infer(tokenizer,model, question, slogan_list=None, temperature=0.1, top_p=None, num_beams=1):
    assert slogan_list is not None
    system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to help users make a purchase decision. The assistant have to choose one product from four options and would not output anyother information."
    slogan_choice ="Here are four products: \n"
    for slogan_idx, slogan in enumerate(slogan_list):
        slogan_choice += f"({slogan_idx+1}) {slogan}\n"
    qs=slogan_choice + question
    messages = [
        {"role": "user", "content": system_prompt+"\n"+qs }
    ]   

    # print(messages)
    encodeds = tokenizer.apply_chat_template(conversation=messages, return_tensors="pt", add_generation_prompt=True)
    
    model_inputs = encodeds.to(model.device)

    # print(model_inputs.shape)
    input_id_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True,
                                    num_beams=num_beams, temperature=temperature, top_p=top_p)
    decoded = tokenizer.batch_decode(generated_ids[:, input_id_length:], skip_special_tokens=True)
    # print(decoded[0])
    # assert False
    return decoded[0]