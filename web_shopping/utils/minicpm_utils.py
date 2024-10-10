import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer




def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, 
                                    torch_dtype=torch.float16,cache_dir="cache",revision="529ee72c846c066cfed6c15e39cb4451f8f8c40e")
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,revision="529ee72c846c066cfed6c15e39cb4451f8f8c40e")
    model.eval()
    return tokenizer, model


def single_infer(tokenizer,model, image_file, question, if_no_figure=False, if_text=False, slogan_list=None, temperature=0.1, top_p=None, num_beams=1):

    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        qs = slogan_choice + question
    else: 
        qs = question
    msgs = [{'role': 'user', 'content': qs}]

    # assert False

    if not if_no_figure:
        ##use figure
        image = Image.open(image_file).convert('RGB')
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, 
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to help users make a purchase decision. The assistant have to choose one product from four options.",
        )
        ##have to add system_prompt otherwise LLM would not give the answer

    else:
        ##no figure text only
        # raise NotImplementedError
        assert if_text == True, "if_text must be True when if_no_figure is True"
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, 
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to help users make a purchase decision. The assistant have to choose one product from four options.",

        )

    return res
