import ast
import csv

# dir_path="output/abstract2concrete"
# output_path="output_csv/final/result.csv"
# average_path="output_csv/final/result_average.csv"


model_list=["Yi-6B-Chat","Yi-VL-6B","Yi-34B-Chat","Yi-VL-34B","Qwen1.5-7B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","qwen-vl-max","qwen-vl-plus","Mistral-7B-Instruct-v0.2","llava-v1.6-mistral-7b-hf", \
    "vicuna-7b-v1.5","llava-v1.6-vicuna-7b-hf","vicuna-13b-v1.5","llava-v1.6-vicuna-13b-hf","Meta-Llama-3-8B-Instruct","cogvlm2-llama3-chat-19B","MiniCPM-Llama3-V-2_5"]



for dataset_type in ["","_extend","_prompt_size","_extend_prompt_size"]:

    dir_path=f"output_prompt/abstract2concrete{dataset_type}"
    output_path=f"output_csv/final/result{dataset_type}.csv"
    average_path=f"output_csv/final/result_average{dataset_type}.csv"
    std_path=f"output_csv/final/result_std{dataset_type}.csv"


    final_result={}
    for model in model_list:
        final_result[model]={}
        final_result[model]["match"]={"big":[],"small":[]}
        final_result[model]["vary"]={"big":[],"small":[]}

    for i in range(1,11):
        file_path=f"{dir_path}/{i}"

        for model in model_list:
            match_path=f"{file_path}/{model}/predictions_match.txt"
            with open(match_path, 'r') as file:
                first_line = file.readline().strip()
                second_line = file.readline().strip()
                big_frequency = ast.literal_eval(first_line.split(': ',1)[1])
                small_frequency = ast.literal_eval(second_line.split(': ',1)[1])
                big=big_frequency.values()
                small=small_frequency.values()

                final_result[model]["match"]["big"].append(big)
                final_result[model]["match"]["small"].append(small)

            vary_path=f"{file_path}/{model}/predictions_vary.txt"
            with open(vary_path, 'r') as file:
                first_line = file.readline().strip()
                second_line = file.readline().strip()
                big_frequency = ast.literal_eval(first_line.split(': ',1)[1])
                small_frequency = ast.literal_eval(second_line.split(': ',1)[1])
                big=big_frequency.values()
                small=small_frequency.values()

                final_result[model]["vary"]["big"].append(big)
                final_result[model]["vary"]["small"].append(small)


    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入标题头
        headers = ['Model', 'Iteration', 'Size', 'Type', '0', '1', '2', 'num_fail']
        writer.writerow(headers)
        
        # 遍历模型
        for model in final_result:
            # 遍历匹配类型
            for match_type in ['match', 'vary']:
                # 遍历大和小
                for size in ['big', 'small']:
                    # 遍历每次迭代
                    for i, freq_list in enumerate(final_result[model][match_type][size]):
                        # 将频率列表转换为可以写入CSV的格式
                        freq_values = list(freq_list)
                        row = [model, i + 1, size.capitalize(), match_type] + freq_values
                        writer.writerow(row)

    ##save the average result for each model
    with open(average_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        headers = ['Model', 'Size','big','middle','small','num_fail_average', '0', '1', '2', 'num_fail_match']
        writer.writerow(headers)

        for model in final_result:

            result={'big':[],'small':[]}
            for match_type in [ 'vary','match']:
                for size in ['big', 'small']:
                    average_freq = [sum(x) / len(x) for x in zip(*final_result[model][match_type][size])]
                    result[size].extend(average_freq)
            for size in ['big', 'small']:
                row = [model, size.capitalize()] + result[size]
                writer.writerow(row)
            

    ##save the std result for each model
    import numpy as np
    with open(std_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        headers = ['Model', 'Size','big','middle','small','num_fail_average', '0', '1', '2', 'num_fail_match']
        writer.writerow(headers)
        
        for model in final_result:

            result={'big':[],'small':[]}
            for match_type in [ 'vary','match']:
                for size in ['big', 'small']:
                    average_freq = [np.std(x) for x in zip(*final_result[model][match_type][size])]
                    result[size].extend(average_freq)
            for size in ['big', 'small']:
                row = [model, size.capitalize()] + result[size]
                writer.writerow(row)
            
                

