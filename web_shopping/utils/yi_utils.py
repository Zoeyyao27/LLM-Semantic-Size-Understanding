import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from llava.conversation import Conversation,SeparatorStyle,conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from enum import Enum, auto






def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_model(model_path):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path,load_8bit=True)
    #left padding
    tokenizer.padding_side = "left"
    return tokenizer, model, image_processor


def single_infer(tokenizer,model,image_processor, image_file, question, if_no_figure=False, if_text=False, slogan_list=None, temperature=0.1, top_p=None, num_beams=1):
    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        if if_no_figure:
            qs=slogan_choice + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + slogan_choice + question
    else: 
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    # assert False

    conv = conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()



    ############################
    if if_text and if_no_figure:
        # text only
        input_ids = tokenizer_image_token(
            prompt, tokenizer, max_length=512, 
            image_token_index=IMAGE_TOKEN_INDEX, 
            return_tensors="pt").unsqueeze(0).cuda()

    else:
        assert if_no_figure is False
        #add the figure
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, max_length=512, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        if not if_no_figure:
            #add the figure
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
            )
        else:
            # raise NotImplementedError("if_text only is not implemented yet")
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
            )    

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    # print(f"outputs:{outputs}")
    return outputs
