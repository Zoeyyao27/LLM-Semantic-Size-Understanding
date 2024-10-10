import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


from time import sleep


from http import HTTPStatus
import dashscope
dashscope.api_key = "sk-16d4b7d6edb14abd8223330bbb8dc2cf"

def simple_multimodal_conversation_call():
    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
                {"text": "这是什么?"}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
                                                     messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.



def single_infer(model_name,image_file, question, if_no_figure=False, if_text=False, slogan_list=None,seed=0, temperature=0.1, top_p=0.8, num_beams=1):

    # if "plus" in model_name:
    question="Which product's name do you find most appealing? Only select one number from 1, 2, 3, or 4. Choose only from the product titles and output only the number corresponding to your choice. Do not include any explanations, reasons, or additional information."
    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        qs = slogan_choice + question
    else:
        assert if_no_figure == False, "if_no_figure must be False when if_text is False"
        qs = question
    # system_prompt = (
    #     "You are acting as a consultant, not a customer. "
    #     "Your task is to choose the most appealing title from four options provided by the user, labeled 1 to 4. "
    #     "You should only output the number of the title that you find most attractive. "
    #     "Do not provide any explanations, reasons, or additional information—simply choose the number."
    # )

    # system_prompt = (
    #     "Pretend to be a human customer making a purchasing decision. "
    #     "Remember this is not a purchase decision, but a simulation. "
    #     "You must choose one product from four options, labeled 1 to 4. "
    #     "Your response must be strictly a single number from 1 to 4, and nothing else. "
    #     "Do not include any explanations, reasons, or additional information. "
    #     "Simply respond with the number corresponding to your choice."
    # )

    system_prompt="You are a shopping assistant. The assistant have to choose one product from four options." \
    "The assistant would only choose one product number from 1 to 4 and would not output anyother information including the product name." \
    "If you found yourself unable to assist with the request, please choose one number from 1 to 4 based on your preference." \
    "Do not include any explanations, reasons, or additional information."\
    "Simply respond with the number corresponding to your choice."

    # print(system_prompt)

    if not if_no_figure:
        ##add figure
        messages = [
            {
                'role': 'system',
                'content': [{
                   "text": system_prompt
                }]
            }, 
            {
                "role": "user",
                "content": [
                    {"image": image_file},
                    {"text": qs}
                ]
            }
        ]
        # print(messages)
        # assert False
    else:
        ##no figure text only
        messages = [
            {
                'role': 'system',
                'content': [{
                    "text": system_prompt
                }]
            }, 
            {
                "role": "user",
                "content": [
                    {"text": qs}
                ]
            }
        ]
        # print(messages)
        # assert False  
    # print(seed)
    # assert False, f"seed:{seed}"
    response = dashscope.MultiModalConversation.call(model=model_name,
                                                     messages=messages,
                                                     temperature=temperature, 
                                                     top_p=top_p,
                                                     max_tokens=512,
                                                     seed=seed)
    if response.status_code == HTTPStatus.OK:
        ans=response["output"]["choices"][0]["message"]["content"][0]["text"]
        print(ans)
        ##wait 3s
        sleep(5)
        return ans
    else:
        print(response.code)
        print(response.message)
        return "Error"
