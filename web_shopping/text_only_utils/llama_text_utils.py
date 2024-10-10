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
    #set padding token to eos token
    return tokenizer, model

                                                 

def single_infer(tokenizer,model, question, slogan_list=None, temperature=0.1, top_p=None, num_beams=1):
    assert slogan_list is not None
    slogan_choice ="Here are four products: \n"
    for slogan_idx, slogan in enumerate(slogan_list):
        slogan_choice += f"({slogan_idx+1}) {slogan}\n"
    qs=slogan_choice + question

    system_prompt="You are a shopping assistant. The assistant have to choose one product from four options." \
    "The assistant would only choose one product number from 1 to 4 and would not output anyother information including the product name." \
    "If you found yourself unable to assist with the request, please choose one number from 1 to 4 based on your preference."

    # print(system_prompt)
    # assert False
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qs}
    ]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to(model.device),
                                max_new_tokens=512, 
                                eos_token_id=terminators,
                                do_sample=True,
                                temperature=temperature, 
                                top_p=top_p,
                                num_beams=num_beams,
                                pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response
