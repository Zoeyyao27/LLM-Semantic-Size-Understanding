import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        cache_dir="cache",
    ).eval()
    return tokenizer, model


def single_infer(tokenizer,model, image_file, question, if_no_figure=False, if_text=False, slogan_list=None, temperature=0.1, top_p=None, num_beams=1):


    ##cogvlm text only template
    #https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B
    # text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
    text_only_template = "USER: {} ASSISTANT:"

    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        qs = slogan_choice + question
    else: 
        qs = question

    # assert False

    if not if_no_figure:
        ##use figure
        image = Image.open(image_file).convert('RGB')
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=qs,
            images=[image],
            template_version='chat'
        )
    else:
        # ##no figure text only
        image = None
        qs=text_only_template.format(qs)
        input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=qs,
                template_version='chat'
            )



    torch_type=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device=model.device
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
    return response

