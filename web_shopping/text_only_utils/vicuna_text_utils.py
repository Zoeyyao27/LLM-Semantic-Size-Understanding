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
    slogan_choice ="Here are four products: \n"
    for slogan_idx, slogan in enumerate(slogan_list):
        slogan_choice += f"({slogan_idx+1}) {slogan}\n"
    qs=slogan_choice + question
    # qs.replace("Question: ","")
    # qs.replace("Answer:","")
    # #replace the question and answer prompt because vicuna does not need them
    input_text = "USER: "+qs+" ASSISTANT:"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids_length=inputs["input_ids"].shape[1]
    outputs = model.generate(inputs["input_ids"].to(model.device), 
                             max_new_tokens=512, 
                             do_sample=True,
                             temperature=temperature, 
                             top_p=top_p,
                             num_beams=num_beams)
    decoded = tokenizer.decode(outputs[0,input_ids_length:], skip_special_tokens=True)
    print(decoded)

    return decoded
