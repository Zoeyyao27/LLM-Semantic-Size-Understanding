
from http import HTTPStatus
import dashscope
import os
import argparse
import torch
import random
import re
import json
import pandas as pd
import time
import tqdm

dashscope.api_key = "sk-16d4b7d6edb14abd8223330bbb8dc2cf"


def label_and_shuffle(items,size_vary):
    if size_vary:
        labels = ['big', 'middle', 'small']
    else:
        labels = ['0', '1', '2']
    items_with_labels = list(zip(items, labels))

    random.shuffle(items_with_labels)

    return {item: label for item, label in items_with_labels}

def read_csv_file(data_path,size_label,if_prompt_size=False):
    ####read big csv file
    data_path=os.path.join(data_path, f"abstract2concrete_{size_label}.csv")
    data = pd.read_csv(data_path, skiprows=1,encoding='utf-8')

    ##clean data
    data = data.replace('\xa0', ' ', regex=True)
    ##replace nan with empty string
    data = data.fillna('')

    final_data=[]
    for index, row in data.iterrows():
        row_list=row.tolist()
        ####abstract big
        question_big=" ".join(row_list[0:3])
        question_small=" ".join(row_list[3:6])

        if "vary" in size_label:
            choices_dict=label_and_shuffle(row_list[6:],True)
        else:
            choices_dict=label_and_shuffle(row_list[6:],False)

        choices_list=list(choices_dict.keys())
        choices_text="(A) "+choices_list[0]+", (B) "+choices_list[1]+", (C) "+choices_list[2] + ". "
        size_list=list(choices_dict.values())

        
        if if_prompt_size:
            question_prompt="Question: Which metaphor do you find most fitting? Consider the semantic size of the given abstract word. "
        else:
            question_prompt="Question: Which metaphor do you find most fitting? "

        ###add big
        temp={}
        temp["input"]= question_prompt + question_big + " [blank]. \nChoices: " + choices_text + "\nAnswer: "
        temp["size_list"]=size_list
        temp["label"]="big"
        final_data.append(temp)

        ###add small
        temp={}
        temp["input"]= question_prompt + question_small + " [blank]. \nChoices: " + choices_text + "\nAnswer: "
        temp["size_list"]=size_list
        temp["label"]="small"
        final_data.append(temp)

    ####################
    return final_data

def data_preprocessing(data_path,args):
    match_data=read_csv_file(data_path,"size_match",args.if_prompt_size)
    vary_data=read_csv_file(data_path,"size_vary",args.if_prompt_size)

    match_dataset = PseudoDataset(match_data)
    vary_dataset = PseudoDataset(vary_data)

    assert len(match_dataset) in [256,110], "Match dataset length is {}, expected 110".format(len(match_dataset))
    return {"match":match_dataset,"vary":vary_dataset}


class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #system_prompt1
        question = "Please fill in the blank with the most fitting metaphor based on your first instinct. "+\
                "Choose one option: (A), (B), or (C). Do not repeat the question or explain your choice. "+\
                "Answer in the format: The answer is (ANSWER CHOICE). If unsure, choose based on your first instinct."

        return {
            'question': question+ self.data[idx]["input"],
            "size_list": self.data[idx]["size_list"],
            'size_label': self.data[idx]["label"],
        }
    
def simple_multimodal_conversation_call(input_text,model_name='qwen-vl-plus'):
    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": input_text}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model=model_name,
                                                     messages=messages)
    if response.status_code == HTTPStatus.OK:
        #print(response)
        ans=response["output"]["choices"][0]["message"]["content"][0]["text"]
        return ans
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.

        return ""

def extract_answer(input,text):

    choices=input.split("Choices: ")[1].split("\n")[0]
    #convert list to dictionary

    # Regular expression to extract choices
    pattern = r"\((\w)\)\s*([^,]*[^,\.\s])"
    matches = re.findall(pattern, choices)

    # Convert matches to dictionary
    choices_dict = {match[1].strip(): match[0] for match in matches}

    ans_text=text

    answer = re.findall(r'\(([A-C])\)', ans_text,flags=re.IGNORECASE)

    if len(answer)>=1:
        # print(answer.group(1))
        # assert False
        if len(answer)>1:
            print("More than one answer found in the text",text)
            print("Answer",answer[0])
        return answer[0], 1
    else:

        ####direct return A,B,C
        letter_text=ans_text.strip().replace("\n","").replace(".","").replace("assistant","").upper()


        if letter_text=="A":
            return "A",1
        elif letter_text=="B":
            return "B",1
        elif letter_text=="C":
            return "C",1
        else:
            
            for i in choices_dict.keys():
                if i.lower() in ans_text.lower():
                    return choices_dict[i],1
            

            print("No answer found in the text",text,choices_dict)
            #assert False
            return random.choice(['A', 'B', 'C']), 0
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen-vl-max')
    parser.add_argument('--data_path', type=str, default="data/abstract2concrete")
    parser.add_argument('--output_dir', type=str, default="output/abstract2concrete")
    parser.add_argument('--if_prompt_size', action='store_true')
    parser.add_argument('--if_extend', action='store_true')
    args = parser.parse_args()

    ##if use extend data, then change the data path
    if args.if_extend:
        args.data_path=args.data_path+"_extend"
        
    ##check if output directory exists
    ouput_dir=os.path.join(args.output_dir,args.model_name.split("/")[-1])
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    
    ##if output_dir is not empty, then ask user if they want to overwrite
    if os.listdir(ouput_dir):
        overwrite=input("Output directory is not empty. Do you want to overwrite it? (y/n)")
        if overwrite.lower()!="y":
            exit()

    match_vary_data_dict = data_preprocessing(args.data_path,args)

    
    for split in match_vary_data_dict: #["match","vary"]

        dataset=match_vary_data_dict[split]

        flag_list=[]
        predictions={"big":[],"small":[]}
        total_output=[]
        total_input=[]
        total_predictions=[]    

        for ins in tqdm.tqdm(dataset):
            ###wait for 1s
            time.sleep(5)
            ##########
            total_input.append(ins["question"])
            ans=simple_multimodal_conversation_call(ins["question"], model_name=args.model_name)
            ans=ans.lower()
            total_output.append(ans)

            answer,flag=extract_answer(ins["question"],ans)
            answer=answer.upper()

            flag_list.append(flag)

            size_list=ins["size_list"] #["big","middle","small"] with random order
            label=ins["size_label"] #big or small
            choice_list=["A","B","C"]

            choose_index=choice_list.index(answer)
            pred_answer=size_list[choose_index]

            if label=="big":
                predictions["big"].append(pred_answer)
            elif label=="small":
                predictions["small"].append(pred_answer)
            else:
                print("Unknown label",label)
                assert False

            total_predictions.append(pred_answer)

        ###big abstract word##########
        big_predictions=predictions["big"]
        print("big_predictions",big_predictions)
        if split=="match":
            big_count=big_predictions.count("0")
            middle_count=big_predictions.count("1")
            small_count=big_predictions.count("2")

        elif split=="vary":
            big_count=big_predictions.count("big")
            middle_count=big_predictions.count("middle")
            small_count=big_predictions.count("small")
        
        else:
            print("Unknown split",split)
            assert False

        total_count=big_count+middle_count+small_count
        
        print("big_count",big_count)
        print("middle_count",middle_count)
        print("small_count",small_count)
        print("total_count",total_count)

        big_frequency={"big":big_count/total_count,"middle":middle_count/total_count,"small":small_count/total_count}
        ##############################

        ###small abstract word##########
        small_predictions=predictions["small"]
        if split=="match":
            big_count=small_predictions.count("0")
            middle_count=small_predictions.count("1")
            small_count=small_predictions.count("2")
        elif split=="vary":
            big_count=small_predictions.count("big")
            middle_count=small_predictions.count("middle")
            small_count=small_predictions.count("small")

        else:
            print("Unknown split",split)
            assert False

        total_count=big_count+middle_count+small_count
        small_frequency={"big":big_count/total_count,"middle":middle_count/total_count,"small":small_count/total_count}

        print("!!!small")
        print("big_count",big_count)
        print("middle_count",middle_count)
        print("small_count",small_count)
        print("total_count",total_count)
  
        ##############################

        ##save the output
        output_path=os.path.join(ouput_dir,f"predictions_{split}.txt")
        with open(output_path, 'w') as f:
            f.write("Big Frequency: "+str(big_frequency)+"\n")
            f.write("Small Frequency: "+str(small_frequency)+"\n")
            f.write("Total output: "+str(total_output)+"\n")
            f.write("Flag list: "+str(flag_list)+"\n")
            f.write("Input: "+str(total_input)+"\n")
            f.write("Predictions: "+str(predictions)+"\n")