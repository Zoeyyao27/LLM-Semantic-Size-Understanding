
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import base64
import requests
import random

# OpenAI API Key
api_key = "your_api_key"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def single_infer(image_file, question,model_name="gpt-4o",
                 if_no_figure=False, if_text=False, slogan_list=None):

    if if_text:
        assert slogan_list is not None
        slogan_choice ="Here are four products: \n"
        for slogan_idx, slogan in enumerate(slogan_list):
            slogan_choice += f"({slogan_idx+1}) {slogan}\n"
        question= slogan_choice + question
    

    if not if_no_figure:
        # Getting the base64 string
        base64_image = encode_image(image_file)

        assert model_name in ["gpt-4o","gpt-4o-turbo","gpt-4o-mini"]
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 10
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    else:
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
                }
            ]
            }
        ],
        "max_tokens": 10
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # print(response.json())
    if response.content:
        try:
            json_data = response.json()
        except ValueError as e:
            print(f"Invalid JSON: {e}")
            print(response)
            #randomly choose one from 4
            answer= random.choice([1,2,3,4])
            print(f"Random answer:{answer}")
            return str(answer)
 
    else:
        print("Empty response received.")
        answer= random.choice([1,2,3,4])
        print(f"Random answer:{answer}")
        return str(answer)

    answer=response.json()["choices"][0]["message"]["content"]
    return answer