import argparse
import os
import json
from collections import Counter
import tqdm
import random
import re

##GLOBAL VARIABLE FAILED
FAILED=0

def clean_result(result):
    # Strip any leading or trailing spaces
    result = result.strip()
    
    # Try to convert the result into an integer
    try:
        result = int(result)
    except ValueError:
        # If conversion fails, check for presence of '1', '2', '3', '4' in the string
        ##find the answer in (ANSWER), return ANSWER
        ans = re.findall(r'(?:\(|\[)(\d)(?:\)|\])', result)
        if len(ans) == 1:
            ans = int(ans[0])
            print(f"!!!find the answer in (ANSWER), return ANSWER:{ans}")
            return ans

        for i in range(1, 5):
            if str(i) in result:
                result = i
                break
        else:
            print(f"no result found in {result}")
            result = random.choice([1, 2, 3, 4])
            ##GLOBAL VARIABLE FAILED
            global FAILED
            FAILED+=1
            print(f"randomly choose one: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/Yi-VL-6B")
    parser.add_argument("--data_path", type=str, default="web_shopping/final_web_dataset/webshop_dataset/screenshots")
    parser.add_argument("--label_path", type=str, default="web_shopping/final_web_dataset/webshop_dataset/total_label.json")
    parser.add_argument("--output_path", type=str, default="output_webshopping")
    parser.add_argument("--if_no_figure", action="store_true")
    parser.add_argument("--if_text", action="store_true")
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    #set random seed
    random.seed(args.random_seed)

    #prepare the output_path
    model_name=args.model_path.split("/")[-1]
    output_path=os.path.join(args.output_path,f"fig_{not args.if_no_figure}_text_{args.if_text}",
                             f"{model_name}",f"{args.random_seed}")
    print(f"output_path:{output_path}")
    result_path=os.path.join(output_path,"result.json")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ##if the output_path is not empty, ask if the user wants to overwrite the file
    if os.listdir(output_path):
        #ask if the user wants to continue
        print(f"output_path {output_path} is not empty")
        print("Do you want to continue? y/n")
        answer=input()
        if answer=="n":
            exit()


    question="Which product would you like to purchase? Only select one number from 1, 2, 3, or 4. Output only the number"

    image_files=[]
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))



    ###########load the total_label############
    with open(args.label_path, 'r') as f:
        total_label=json.load(f)
    slogan_dict={} #save the slogan list for each product
    for i in total_label:
        slogan_dict[i]=[]
        for j in total_label[i]["slogan"]:
            slogan_dict[i].append(j["product"]["name"])
    
    ##load model
    #text only models
    if args.model_path=="mistralai/Mistral-7B-Instruct-v0.2":
        from web_shopping.text_only_utils.mistral_text_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif "lmsys/vicuna" in args.model_path:
        from web_shopping.text_only_utils.vicuna_text_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif args.model_path in ["01-ai/Yi-6B-Chat","01-ai/Yi-34B-Chat"]:
        from web_shopping.text_only_utils.yi_text_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif args.model_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        from web_shopping.text_only_utils.llama_text_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif args.model_path in ["Qwen/Qwen-7B-Chat","Qwen/Qwen1.5-7B-Chat"]:
        from web_shopping.text_only_utils.qwen_text_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    ##VL models
    elif "Yi" in args.model_path:
        from web_shopping.utils.yi_utils import single_infer,load_model
        tokenizer, model, image_processor=load_model(args.model_path)
    elif "llava" in args.model_path:
        from web_shopping.utils.lava_utils import single_infer,load_model
        tokenizer, model, processor=load_model(args.model_path)
    elif "gpt" in args.model_path: 
        from web_shopping.utils.gpt4o_utils import single_infer
        tokenizer=None
        model=None
        image_processor=None
    elif "cogvlm" in args.model_path:
        from web_shopping.utils.cogvlm_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif "MiniCPM" in args.model_path:
        from web_shopping.utils.minicpm_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif args.model_path == "Qwen/Qwen-VL-Chat":
        from web_shopping.utils.qwen_vl_utils import single_infer,load_model
        tokenizer, model=load_model(args.model_path)
    elif args.model_path in ["qwen-vl-plus","qwen-vl-max"]:
        from web_shopping.utils.qwen_api_utils import single_infer


    final_results={}
    for image_file in tqdm.tqdm(image_files):
        product_id=int(image_file.split("/")[-1].split(".")[0].split("_")[-1])
        slogan_list=slogan_dict[str(product_id)]
        if args.model_path=="mistralai/Mistral-7B-Instruct-v0.2" \
            or "lmsys/vicuna" in args.model_path \
            or args.model_path in ["01-ai/Yi-6B-Chat","01-ai/Yi-34B-Chat"]\
            or args.model_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            assert args.if_no_figure==True, f"{args.model_path}only support text only"
            result= single_infer(tokenizer, model, question,slogan_list=slogan_list)
        elif args.model_path in ["Qwen/Qwen-7B-Chat","Qwen/Qwen1.5-7B-Chat"]:
            assert args.if_no_figure==True, f"{args.model_path}only support text only"
            result= single_infer(tokenizer, model, args.model_path, question, slogan_list=slogan_list)
        elif "Yi" in args.model_path:
            result= single_infer(tokenizer, model, image_processor, image_file, question,
                                 if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif "llava" in args.model_path:
            result=single_infer(tokenizer, model, processor, image_file,question,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif "gpt" in args.model_path:
            result=single_infer(image_file, question,args.model_path,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif "cogvlm" in args.model_path:
            result=single_infer(tokenizer, model, image_file, question,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif "MiniCPM" in args.model_path:
            result=single_infer(tokenizer, model, image_file, question,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif args.model_path == "Qwen/Qwen-VL-Chat":
            result=single_infer(tokenizer, model, image_file, question,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list)
        elif args.model_path in ["qwen-vl-plus","qwen-vl-max"]:
            result=single_infer(args.model_path,image_file,question,
                                if_no_figure=args.if_no_figure,if_text=args.if_text,slogan_list=slogan_list,seed=args.random_seed)
            
        result=clean_result(result)
        assert product_id not in final_results
        final_results[product_id]=result
    

    
    #calculate the accuracy    
    chosen_keys=[]
    chosen_key_dict={}
    for i in final_results:
        if final_results[i]-1 not in [0,1,2,3]:
            print(f"final_results[{i}]:{final_results[i]}")
            print(f"total_label[{i}]:{total_label[str(i)]}")
            #ramdomly choose one
            final_results[i]=random.choice([1,2,3,4])
            
        chosen_keys.append(total_label[str(i)]["label"][final_results[i]-1])  #label in the picture start from 1
        chosen_key_dict[i]=total_label[str(i)]["label"][final_results[i]-1]

    ##save the results
    with open(result_path, 'w') as f:
        json.dump(final_results,f)
        #add the chosen_keys_dict
        f.write("\n")
        json.dump(chosen_key_dict,f)
        
    
    #calculate the probability of the chosen_keys
    chosen_keys_counter=Counter(chosen_keys)
    chosen_keys_prob={k:chosen_keys_counter[k]/len(chosen_keys) for k in chosen_keys_counter}
    print(chosen_keys_prob)
    #save the chosen_keys_prob
    chosen_keys_prob_path=os.path.join(output_path,"chosen_keys_prob.json")
    with open(chosen_keys_prob_path, 'w') as f:
        json.dump(chosen_keys_prob,f)
        #DUMP THE FAILED
        f.write("\n")
        json.dump({"FAILED":FAILED},f)



    
