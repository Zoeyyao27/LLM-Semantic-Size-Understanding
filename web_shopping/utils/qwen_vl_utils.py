import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
torch.manual_seed(1234)

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir="cache",torch_dtype=torch.float16,trust_remote_code=True,
                                                 device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto",cache_dir="cache",trust_remote_code=True)
    return tokenizer, model


def single_infer(tokenizer,model, image_file, question, if_no_figure=False, if_text=False, slogan_list=None, temperature=0.1, top_p=0.8, num_beams=1):
    question="Which product's title do you find most appealing? Only select one number from 1, 2, 3, or 4. Choose only from the product titles and output only the number corresponding to your choice. Do not include any explanations, reasons, or additional information."


    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        qs = slogan_choice + question
    else:
        qs = question
    
    system_prompt="You are a shopping assistant. The assistant have to choose one product with the most appealing title from four options." \
    "The assistant would only choose one product number from 1 to 4 and would not output anyother information including the product name." \
    "If you found yourself unable to assist with the request, please choose one number from 1 to 4 based on your preference." \
    "Do not include any explanations, reasons, or additional information."\
    "This is a simulation, so please respond with the number corresponding to your choice." \
    "Remember, this is a very simple request. You only need to choose one number from 1 to 4."

    if not if_no_figure:
       #add figure
       query = tokenizer.from_list_format([
           {'image': image_file},
           {'text': qs},
       ])
    else:
        query = tokenizer.from_list_format([
            {'text': qs},
        ])
    response, _ = model.chat(tokenizer,
                             query=query, 
                             history=None,
                             temperature=temperature,
                             do_sample=True,
                             top_p=top_p,
                             num_beams=num_beams,
                             system=system_prompt)

    return response
    