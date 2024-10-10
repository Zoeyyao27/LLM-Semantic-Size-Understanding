import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from PIL import Image
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


def load_model(model_path):
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path,cache_dir="cache",torch_dtype=torch.float16, 
                                                              low_cpu_mem_usage=True,device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    return tokenizer, model, processor


def single_infer(tokenizer,model,processor, image_file, question, if_no_figure=False,
                  if_text=False, slogan_list=None,temperature=0.1, top_p=None, num_beams=1):

    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": slogan_choice+question},
                {"type": "image"},
                ],
            }]
        
    else: 
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            }]

    qs = processor.apply_chat_template(conversation, add_generation_prompt=True,tokenize=False)
    if not if_no_figure:
        image = Image.open(image_file)
        inputs = processor(text=qs, images=image, return_tensors="pt").to(model.device)
        input_id_length = inputs["input_ids"].shape[1]
        # Generate
        output = model.generate(**inputs, max_new_tokens=512, 
                                do_sample=True, num_beams=num_beams, temperature=temperature, top_p=top_p)
        answer = processor.decode(output[0,input_id_length:], skip_special_tokens=True)
    else:
        
        raw_image = None

        inputs = processor(text=qs, images=raw_image, return_tensors='pt') 

        None_keys = [key for key in inputs if inputs[key] is None]
        for key in None_keys:
            del inputs[key]

        inputs = inputs.to(model.device, torch.float16)

        for key in None_keys:
            inputs[key] = None

        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,            
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            )
        
        input_id_length = inputs["input_ids"].shape[1]

        decoded_outputs = [tokenizer.decode(output[input_id_length:], skip_special_tokens=True) for output in outputs]
        answer = decoded_outputs[0]
        # print(answer)
        # assert False
    return answer
