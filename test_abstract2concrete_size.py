from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, LlavaForConditionalGeneration,Qwen2VLForConditionalGeneration

import argparse
import os
import pandas as pd
import random  
import re
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
    tokenizer_image_token_with_mask,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from llava.conversation import Conversation, SeparatorStyle

from qwen_vl_utils import process_vision_info



class Abstract2ConcreteDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.system_prompt = "Please fill in the blank with the most fitting metaphor based on your first instinct. "+\
                "Choose one option: (A), (B), or (C). Do not repeat the question or explain your choice. "+\
                "Answer in the format: The answer is (ANSWER CHOICE). If unsure, choose based on your first instinct."
 
        if "Yi" in self.tokenizer.name_or_path and "VL" in self.tokenizer.name_or_path:
            mm_default_conv = Conversation(
                system= self.system_prompt,
                roles=("Human", "Assistant"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.SINGLE,
                sep="###",
            )

            self.conv_templates = {
                "mm_default": mm_default_conv,
            }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        # Tokenize the input text
        if "llava" in self.tokenizer.name_or_path:
            input_dict = {}
            #input_text = self.system_prompt + "\n" + input_text 
            # \nAnswer:  and \nQuestion:
            input_text = input_text.replace("\nAnswer: ","")
            input_text = input_text.replace("Question:","")
            input_text=self.system_prompt+" USER:"+input_text+"\nASSISTANT:"
            # print("Input text",input_text)
            # assert False
            if "mistral" in self.tokenizer.name_or_path:
                input_text = f"[INST] {input_text} [/INST]"
            input_dict["size_list"]=item["size_list"]
            input_dict["size_label"]=item["label"]
            input_dict["text"]=input_text
            return input_dict
        
        if "MiniCPM-Llama3-V-2_5" in self.tokenizer.name_or_path:
            
            input_dict={}
            input_dict["system_prompt"]=self.system_prompt
            input_dict["size_list"]=item["size_list"]
            input_dict["size_label"]=item["label"]
            input_dict["input_text"]= input_text
            return input_dict
        if self.tokenizer.name_or_path == "Qwen/Qwen2-VL-7B-Instruct":
            input_dict={}
            input_dict["system_prompt"]=self.system_prompt
            input_dict["size_list"]=item["size_list"]
            input_dict["size_label"]=item["label"]
            input_dict["input_text"]= input_text
            return input_dict



    
        
        if "Chat" in self.tokenizer.name_or_path and "Yi" in self.tokenizer.name_or_path:

            messages = [
                {
                    "role": "system", 
                    "content":  self.system_prompt ,
                 },
                {"role": "user", "content": input_text }
                 #input_text }
            ]       

            input_text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
            #print("Input text",input_text)  

        elif "Qwen" in self.tokenizer.name_or_path and "Chat" in self.tokenizer.name_or_path:
            
            messages = [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {"role": "user", "content": input_text }
            ]
                    

            input_text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        elif "Qwen2" in self.tokenizer.name_or_path:
            
            messages = [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {"role": "user", "content": input_text }
            ]
                    

            input_text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        elif "vicuna" in self.tokenizer.name_or_path:
                    

            input_text = self.system_prompt+" USER: "+input_text+" ASSISTANT: "

        elif "Mistral" in self.tokenizer.name_or_path:
            if "Instruct" in self.tokenizer.name_or_path:
                messages = [
                    {
                        "role": "user", "content": self.system_prompt+ "\n"+input_text }
                ]   

                input_text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
                # print("Input text",input_text)
            else:
                input_text = self.system_prompt+"\n"+input_text+" "

        elif "Yi" in self.tokenizer.name_or_path and "VL" in self.tokenizer.name_or_path:
            conv = self.conv_templates["mm_default"].copy()
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], None)
            input_text = conv.get_prompt()
            
            input_ids, attention_mask = tokenizer_image_token_with_mask(input_text, tokenizer, self.max_length, IMAGE_TOKEN_INDEX, return_tensors="pt")

            res={"input_ids":input_ids,"attention_mask":attention_mask}
            res["size_list"]=item["size_list"]
            res["size_label"]=item["label"]
            # print("input_ids",input_ids.shape)
            # assert False
            return res
        
        elif "cogvlm2" in self.tokenizer.name_or_path:
            input_text = f"{self.system_prompt} USER: {input_text} ASSISTANT:"

        elif "Llama-3" in self.tokenizer.name_or_path:
            messages = [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {"role": "user", "content": input_text }
            ]
            input_text = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
            )


        elif "llama" in self.tokenizer.name_or_path:
            # assert False, "Not implemented"
            base_prompt = "<s>[INST]\n<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
            input_text = base_prompt.format(system_prompt="Please complete the following sentence by filling the blank based on your first instinct and choose either (A), (B), or (C) as your answer. Do not provide reason.", user_prompt=input_text)
        else:
            
            input_text = "Please complete the following sentence by filling the blank based on your first instinct and choose either (A), (B), or (C) as your answer. " + input_text

            assert False


        input_ids = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=self.max_length, padding="max_length")
        input_ids["size_list"]=item["size_list"]
        input_ids["size_label"]=item["label"]

        if "cogvlm2" in self.tokenizer.name_or_path:
            #cogvlm2 need to add token_type_ids
            LANGUAGE_TOKEN_TYPE = 0
            token_type_ids=[LANGUAGE_TOKEN_TYPE] * len(input_ids["input_ids"][0])
            input_ids['token_type_ids']=torch.tensor(token_type_ids, dtype=torch.long),
        return input_ids
    




def label_and_shuffle(items,size_vary):
    if size_vary:
        labels = ['big', 'middle', 'small']
    else:
        labels = ['0', '1', '2']

    items_with_labels = list(zip(items, labels))

    random.shuffle(items_with_labels)
    return {item: label for item, label in items_with_labels}

def read_csv_file(data_path,size_label,if_prompt_size=False,if_add_q_a=True):
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
            if if_add_q_a:
                question_prompt="Question: Which metaphor do you find most fitting? Consider the semantic size of the given abstract word. "
            else:
                question_prompt="Which metaphor do you find most fitting? Consider the semantic size of the given abstract word. "
        else:
            if if_add_q_a:
                #assert False, "Not implemented"
                question_prompt="Question: Which metaphor do you find most fitting? "
            else:
                question_prompt="Which metaphor do you find most fitting? "

        ###add big
        temp={}
        if if_add_q_a:
            temp["input"]= question_prompt + question_big + " [blank]. \nChoices: " + choices_text + "\nAnswer: "
        else:
            temp["input"]= question_prompt + question_big + " [blank]. \nChoices: " + choices_text

        temp["size_list"]=size_list
        temp["label"]="big"
        final_data.append(temp)

        ###add small
        temp={}
        if if_add_q_a:
            temp["input"]= question_prompt + question_small + " [blank]. \nChoices: " + choices_text + "\nAnswer: "
        else:
            temp["input"]= question_prompt + question_small + " [blank]. \nChoices: " + choices_text

        temp["size_list"]=size_list
        temp["label"]="small"
        final_data.append(temp)

    ####################
    return final_data

def data_preprocessing(data_path,tokenizer,max_length,args):
    match_data=read_csv_file(data_path,"size_match",args.if_prompt_size,args.if_add_q_a)
    vary_data=read_csv_file(data_path,"size_vary",args.if_prompt_size,args.if_add_q_a)

    match_dataset = Abstract2ConcreteDataset(match_data, tokenizer=tokenizer, max_length=max_length)
    vary_dataset = Abstract2ConcreteDataset(vary_data, tokenizer=tokenizer, max_length=max_length)

    assert len(match_dataset) in [256,110], "Match dataset length is {}, expected 110".format(len(match_dataset))
    return match_dataset,vary_dataset

def extract_answer(text,if_add_q_a=True):
    # print(text)
    #######################################################
    choices=text.split("Choices: ")[1].split("\n")[0]
    #convert list to dictionary

    # Regular expression to extract choices
    pattern = r"\((\w)\)\s*([^,]*[^,\.\s])"
    matches = re.findall(pattern, choices)

    # Convert matches to dictionary
    choices_dict = {match[1].strip(): match[0] for match in matches}
    #######################################################

    if if_add_q_a:
        if "Answer: " in text:
            ans_text="".join(text.split("Answer: ")[1:])
        else:
            ans_text="".join(text.split("Answer:")[1:])
    else:
        if "assistant" in text:
            ans_text="".join(text.split("assistant")[1:])
        elif "Assistant: " in text:
            ans_text="".join(text.split("Assistant: ")[1:])
        elif "ASSISTANT" in text:
            ans_text="".join(text.split("ASSISTANT")[1:])
        elif "[/INST]" in text:
            ans_text="".join(text.split("[/INST]")[1:]) #mistral when not add question and answer
        else:
            raise ValueError("No assistant found in the text",text)
            #print("No assistant found in the text",text)

    # if "Assistant: " in text:
    #     ans_text="".join(text.split("Assistant: ")[1:])

    answer = re.findall(r'\(([A-C])\)', ans_text)
    if len(answer)>=1:
        if len(answer)>1:
            print("More than one answer found in the text",text)
            print("Answer",answer[0])
        return answer[0], 1
    else:

        ####direct return A,B,C
        letter_text=ans_text.strip().replace("\n","").replace(".","").replace("assistant","").replace("###","").upper()

        letter_text=letter_text.strip()

        if letter_text=="A":
            return "A",1
        elif letter_text=="B":
            return "B",1
        elif letter_text=="C":
            return "C",1
        else:
            
            print("!!!choices_list",choices_dict,ans_text)
            for i in choices_dict.keys():
                if i.lower() in ans_text.lower():
                    print("Answer found in the text",text)
                    print("Match",choices_dict[i])
                    return choices_dict[i],1
            
            print("No answer found in the text",text,choices_dict)
            #assert False
            return random.choice(['A', 'B', 'C']), 0
    
def generate_predictions(model, dataloader,tokenizer,if_add_q_a=True):
    model.eval()  
    device=model.device
    predictions = {"big": [], "small": []}
    total_output=[]
    flags=[]

    with torch.no_grad():  
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device).squeeze(1)
            attention_mask = batch['attention_mask'].to(device).squeeze(1)
            if "cogvlm2" in model.name_or_path:
                #cogvlm2 need to add token_type_ids
                token_type_ids=batch['token_type_ids'][0].to(device).squeeze(1)

            
                # generate predictons
                outputs = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    max_new_tokens=128,            
                    do_sample=False,
                    num_beams=1
                    )
            elif "Llama-3" in model.name_or_path:
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    eos_token_id=terminators,
                    do_sample=False,
                    num_beams=1,
                )
            else:
                # generate predictons
                outputs = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    max_new_tokens=128,            
                    do_sample=False,
                    num_beams=1
                    )
            
            

            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            total_output+=decoded_outputs
            answer=[]
            
            
            for output in decoded_outputs:
                ans,flag=extract_answer(output,if_add_q_a=if_add_q_a)
                answer.append(ans)
                flags.append(flag)

            size_list = [list(t) for t in zip(*batch["size_list"])]
                
            label=batch["size_label"]


            choice_list=["A","B","C"]
            for ans,size_list_item,label in zip(answer,size_list,label):
                choose_idx=choice_list.index(ans)
                pred_answer=size_list_item[choose_idx]
                if label=="big":
                    predictions["big"].append(pred_answer)
                elif label=="small":
                    predictions["small"].append(pred_answer)
                else:
                    print("Label is not big or small")
                    assert False

    return predictions,flags,total_output

    
def generate_predictions_llava(model,processor, dataloader,if_add_q_a=True):
    model.eval()  
    device=model.device
    predictions = {"big": [], "small": []}
    total_output=[]
    flags=[]

    with torch.no_grad():  
        for batch in tqdm.tqdm(dataloader):
            raw_image = None

            inputs = processor(batch["text"], raw_image, return_tensors='pt') 

            None_keys = [key for key in inputs if inputs[key] is None]

            for key in None_keys:
                del inputs[key]

            inputs = inputs.to(device, torch.float16)
            for key in None_keys:
                inputs[key] = None

            outputs = model.generate(
                **inputs, 
                max_new_tokens=128,            
                do_sample=False,
                num_beams=1
                )


            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            total_output+=decoded_outputs
            answer=[]
            
            
            for output in decoded_outputs:

                ans,flag=extract_answer(output,if_add_q_a=if_add_q_a)
                answer.append(ans)
                flags.append(flag)

            size_list = [list(t) for t in zip(*batch["size_list"])]
                
            label=batch["size_label"]


            choice_list=["A","B","C"]
            for ans,size_list_item,label in zip(answer,size_list,label):
                choose_idx=choice_list.index(ans)
                pred_answer=size_list_item[choose_idx]
                # print("Predicted answer",pred_answer)
                # assert False
                if label=="big":
                    predictions["big"].append(pred_answer)
                elif label=="small":
                    predictions["small"].append(pred_answer)
                else:
                    print("Label is not big or small")
                    assert False

    return predictions,flags,total_output

     
def generate_predictions_qwen2_vl(model, dataloader,processor,if_add_q_a=True):
    model.eval()  
    device=model.device
    predictions = {"big": [], "small": []}
    total_output=[]
    flags=[]

    with torch.no_grad():  
        for batch in tqdm.tqdm(dataloader):
            #batch_size=1
            input_text=batch["input_text"][0]
            system_prompt=batch["system_prompt"][0]
            assert len(batch["input_text"])==1, "Only support batch size 1 for now, but got {}".format(len(batch["input_text"]))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs=inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )


            decoded_outputs = [input_text+output_text[0]]
            
            total_output+=decoded_outputs
            answer=[]
            
            
            for output in decoded_outputs:
                ans,flag=extract_answer(output,if_add_q_a=if_add_q_a)
                answer.append(ans)
                flags.append(flag)

            size_list = [list(t) for t in zip(*batch["size_list"])]
                
            label=batch["size_label"]


            choice_list=["A","B","C"]
            for ans,size_list_item,label in zip(answer,size_list,label):
                choose_idx=choice_list.index(ans)
                pred_answer=size_list_item[choose_idx]
                if label=="big":
                    predictions["big"].append(pred_answer)
                elif label=="small":
                    predictions["small"].append(pred_answer)
                else:
                    print("Label is not big or small")
                    assert False

    return predictions,flags,total_output
   
def generate_predictions_yi_vl(model, dataloader,tokenizer,if_add_q_a=True):
    model.eval()  
    device=model.device
    predictions = {"big": [], "small": []}
    total_output=[]
    flags=[]

    keywords = ["###"]


    with torch.no_grad():  
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"].to(device).squeeze(1)
            attention_mask = batch["attention_mask"].to(device).squeeze(1)
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            

            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=None,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=128,            
                do_sample=False,
            )

            decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            total_output+=decoded_outputs

            answer=[]
            
            
            for output in decoded_outputs:
                output=output.strip()
                if output.endswith(keywords[0]):
                    output = output[: -len(keywords[0])]
                ans,flag=extract_answer(output,if_add_q_a=if_add_q_a)
                answer.append(ans)
                flags.append(flag)

            size_list = [list(t) for t in zip(*batch["size_list"])]
                
            label=batch["size_label"]


            choice_list=["A","B","C"]
            for ans,size_list_item,label in zip(answer,size_list,label):
                choose_idx=choice_list.index(ans)
                pred_answer=size_list_item[choose_idx]

                if label=="big":
                    predictions["big"].append(pred_answer)
                elif label=="small":
                    predictions["small"].append(pred_answer)
                else:
                    print("Label is not big or small")
                    assert False

    return predictions,flags,total_output


def generate_predictions_MiniCPM(model, dataloader,if_add_q_a=True):
    model.eval()  
    device=model.device
    predictions = {"big": [], "small": []}
    total_output=[]
    flags=[]

    with torch.no_grad():  
        for batch in tqdm.tqdm(dataloader):
            #batch_size=1
            input_text=batch["input_text"][0]
            system_prompt=batch["system_prompt"][0]
            assert len(batch["input_text"])==1, "Only support batch size 1 for now, but got {}".format(len(batch["input_text"]))

            msgs=[{"role":"user","content":input_text}]

            outputs = model.chat(
                image=None,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=False,
                system_prompt=system_prompt
            )


            decoded_outputs = [input_text+outputs]
            
            total_output+=decoded_outputs

            answer=[]
            
            
            for output in decoded_outputs:
                ans,flag=extract_answer(output,if_add_q_a=if_add_q_a)
                answer.append(ans)
                flags.append(flag)

            size_list = [list(t) for t in zip(*batch["size_list"])]
                
            label=batch["size_label"]


            choice_list=["A","B","C"]
            for ans,size_list_item,label in zip(answer,size_list,label):
                choose_idx=choice_list.index(ans)
                pred_answer=size_list_item[choose_idx]
                if label=="big":
                    predictions["big"].append(pred_answer)
                elif label=="small":
                    predictions["small"].append(pred_answer)
                else:
                    print("Label is not big or small")
                    assert False

    return predictions,flags,total_output

           


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen-7B-Chat") #"01-ai/Yi-6B""meta-llama/Llama-2-13b-chat-hf" "01-ai/Yi-6B-Chat" ""
    parser.add_argument('--data_path', type=str, default="data/abstract2concrete")
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--output_dir', type=str, default="output/abstract2concrete")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--if_prompt_size', action='store_true')
    # parser.add_argument('--load_in_8bit', action='store_false') #load in 8bit
    parser.add_argument('--if_add_q_a', action='store_false')
    parser.add_argument('--if_extend', action='store_true')
    args = parser.parse_args()

    # assert args.load_in_8bit==True, "Only support 8bit for now"

    ##if use extend data, then change the data path
    if args.if_extend:
        args.data_path=args.data_path+"_extend"

    ###llava-vicuna set add_qa to false, add qa will cause performance drop
    if "vicuna" in args.model_name and "llava" in args.model_name:
        args.if_add_q_a=False

    ##check if output directory exists
    ouput_dir=os.path.join(args.output_dir,args.model_name.split("/")[-1])
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    
    ##if output_dir is not empty, then ask user if they want to overwrite
    if os.listdir(ouput_dir):
        overwrite=input("Output directory is not empty. Do you want to overwrite it? (y/n)")
        if overwrite.lower()!="y":
            exit()

    #############load model##############
    if "Qwen" in args.model_name:
        if args.model_name=="Qwen/Qwen2-VL-7B-Instruct":
            # default: Load the model on the available device(s)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )

            # default processer
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            tokenizer = processor.tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map='auto', trust_remote_code=True, cache_dir=args.cache_dir).eval()

            tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir,
                                                    trust_remote_code=True)
        
        if "Qwen2" not in args.model_name:
            tokenizer.eos_token='<|endoftext|>'

    elif "Yi" and "VL" in args.model_name:
        model_path = os.path.expanduser(args.model_name)
        key_info["model_path"] = model_path
        tokenizer, base_model, _, _ = load_pretrained_model(args.model_name)
        model = base_model.eval()

    elif "llava" in args.model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            device_map="auto",
            cache_dir=args.cache_dir,
        )

        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer = processor.tokenizer
    elif "cogvlm2" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map='auto',
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        ).eval()
    elif "MiniCPM" in args.model_name:
        #use revision to fix the model, newesr version has bug when text-only generation
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,device_map="auto",cache_dir=args.cache_dir, trust_remote_code=True,revision="529ee72c846c066cfed6c15e39cb4451f8f8c40e")
        model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto",cache_dir=args.cache_dir, trust_remote_code=True,revision="529ee72c846c066cfed6c15e39cb4451f8f8c40e").eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,device_map="auto",cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto",cache_dir=args.cache_dir).eval()

        


    if "llava" not in args.model_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Tokenizer padding token",tokenizer.pad_token)
        tokenizer.padding_side = "left"

        if "Qwen/Qwen-7B-Chat" in args.model_name or "Mistral" in args.model_name \
            or "MiniCPM-Llama3-V-2_5" in args.model_name or "Qwen2" in args.model_name:
            #fix batch size to 1 due to weird error in Qwen-7B-Chat model when batch inference
            #TO DO: https://github.com/QwenLM/Qwen/issues/414
            args.batch_size=1
        match_data,vary_data = data_preprocessing(args.data_path,tokenizer,args.max_length,args)
        match_dataloader = DataLoader(match_data, batch_size=args.batch_size)
        vary_dataloader = DataLoader(vary_data, batch_size=args.batch_size)

        if "Yi" in args.model_name and "VL" in args.model_name:
            match_predictions,match_flags,total_output_match = generate_predictions_yi_vl(model, match_dataloader,tokenizer,args.if_add_q_a)
            vary_predictions,vary_flags,total_output_vary= generate_predictions_yi_vl(model, vary_dataloader,tokenizer,args.if_add_q_a)
        elif args.model_name=="Qwen/Qwen2-VL-7B-Instruct":
            match_predictions,match_flags,total_output_match = generate_predictions_qwen2_vl(model, match_dataloader,processor,args.if_add_q_a)
            vary_predictions,vary_flags,total_output_vary= generate_predictions_qwen2_vl(model, vary_dataloader,processor,args.if_add_q_a)

        elif "MiniCPM" in args.model_name:
            match_predictions,match_flags,total_output_match = generate_predictions_MiniCPM(model, match_dataloader,args.if_add_q_a)
            vary_predictions,vary_flags,total_output_vary= generate_predictions_MiniCPM(model, vary_dataloader,args.if_add_q_a)
        else:
            match_predictions,match_flags,total_output_match = generate_predictions(model, match_dataloader,tokenizer,args.if_add_q_a)
            vary_predictions,vary_flags,total_output_vary= generate_predictions(model, vary_dataloader,tokenizer,args.if_add_q_a)
    
    else:
        args.batch_size=1
        match_data,vary_data = data_preprocessing(args.data_path,tokenizer,args.max_length,args)
        match_dataloader = DataLoader(match_data, batch_size=args.batch_size)
        vary_dataloader = DataLoader(vary_data, batch_size=args.batch_size)

        match_predictions,match_flags,total_output_match = generate_predictions_llava(model,processor, match_dataloader,args.if_add_q_a)
        vary_predictions,vary_flags,total_output_vary= generate_predictions_llava(model,processor, vary_dataloader,args.if_add_q_a)
    


    ### size match predictions##############
    big_predictions=match_predictions["big"]
    small_predictions=match_predictions["small"]

    ####big
    big_count=big_predictions.count("0")
    middle_count=big_predictions.count("1")
    small_count=big_predictions.count("2")
    total_count=big_count+middle_count+small_count
    big_frequency_match={"0":big_count/total_count,"1":middle_count/total_count,"2":small_count/total_count,"num_fail":len(match_flags)-sum(match_flags)}

    #####small
    big_count=small_predictions.count("0")
    middle_count=small_predictions.count("1")
    small_count=small_predictions.count("2")
    total_count=big_count+middle_count+small_count
    small_frequency_match={"0":big_count/total_count,"1":middle_count/total_count,"2":small_count/total_count,"num_fail":len(match_flags)-sum(match_flags)}
    ########################################

    ### size vary predictions##############
    big_predictions=vary_predictions["big"]
    small_predictions=vary_predictions["small"]

    ####big
    big_count=big_predictions.count("big")
    middle_count=big_predictions.count("middle")
    small_count=big_predictions.count("small")
    total_count=big_count+middle_count+small_count
    big_frequency_vary={"big":big_count/total_count,"middle":middle_count/total_count,"small":small_count/total_count,"num_fail":len(vary_flags)-sum(vary_flags)}

    #####small
    big_count=small_predictions.count("big")
    middle_count=small_predictions.count("middle")
    small_count=small_predictions.count("small")
    total_count=big_count+middle_count+small_count
    small_frequency_vary={"big":big_count/total_count,"middle":middle_count/total_count,"small":small_count/total_count,"num_fail":len(vary_flags)-sum(vary_flags)}
    ########################################

    print("Match frequency")
    print("!!!big",big_frequency_match)
    print("!!!small",small_frequency_match)

    print("Vary frequency")
    print("!!!big",big_frequency_vary)
    print("!!!small",small_frequency_vary)

    ##save the results
    with open(os.path.join(ouput_dir,"predictions_match.txt"), "w") as f:
        f.write("Big frequency: "+str(big_frequency_match)+"\n")
        f.write("Small frequency: "+str(small_frequency_match)+"\n")
        f.write("predictions: "+str(match_predictions)+"\n")
        f.write("flags: "+str(match_flags)+"\n")
        f.write("total_output: "+str(total_output_match)+"\n")

    with open(os.path.join(ouput_dir,"predictions_vary.txt"), "w") as f:
        f.write("Big frequency: "+str(big_frequency_vary)+"\n")
        f.write("Small frequency: "+str(small_frequency_vary)+"\n")
        f.write("predictions: "+str(vary_predictions)+"\n")
        f.write("flags: "+str(vary_flags)+"\n")
        f.write("total_output: "+str(total_output_vary)+"\n")