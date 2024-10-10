import os
import json
import csv
import statistics

file_path="output_webshopping"
output_path=os.path.join(file_path,"output_avergae")
if not os.path.exists(output_path):
    os.makedirs(output_path)

for if_figure in [True,False]:
    for if_text in [True,False]:
        if not if_figure and not if_text:
            continue
        dir_path=os.path.join(file_path,f"fig_{if_figure}_text_{if_text}")
        #遍历文件夹下所有的model name的文件夹
        final_result={}
        for model_name in os.listdir(dir_path):
            model_path=os.path.join(dir_path,model_name)
            final_result[model_name]={}
            for seed in os.listdir(model_path):
                seed_path=os.path.join(model_path,seed)
                final_result[model_name][seed]={}
                for file in os.listdir(seed_path):
                    if file.endswith(".json") and file.startswith("chosen_keys_prob"):
                        chosen_keys_prob_path=os.path.join(seed_path,file)
                        with open(chosen_keys_prob_path,"r") as f:
                            total_data=[]
                            for line in f:
                                data=json.loads(line)
                                total_data.append(data)
                            if len(total_data)==1:
                                final_result[model_name][seed]=total_data[0]
                                final_result[model_name][seed]["failed"]=0
                            elif len(total_data)>1:
                                final_result[model_name][seed]=total_data[0]
                                final_result[model_name][seed]["failed"]=total_data[1]["FAILED"]

        ##save the final_result to a csv file
        # 获取所有的模型名称和键名列表
        models = list(final_result.keys())
        keys = sorted({int(k) for model_data in final_result.values() for k in model_data.keys()})
        filename = os.path.join(output_path, f"average_if_figure_{if_figure}_if_text_{if_text}.csv")

        # 创建CSV文件并写入数据
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 写入列标题行
            header_row_1 = [""] + ["semantic"] * (len(keys) + 1) + ["eye_catch"] * (len(keys) + 1) + ["polish"] * (len(keys) + 1) + ["original"] * (len(keys) + 1) + ["Failed"] * (len(keys) + 1)
            header_row_2 = [""] + keys + ["Average"] + keys + ["Average"] + keys + ["Average"] + keys + ["Average"]
            writer.writerow(header_row_1)
            writer.writerow(header_row_2)

            # 写入每个模型的数据
            for model in models:
                row = [model]
                
                # 处理 semantic 列
                semantic_values = [final_result[model].get(str(key), {}).get("semantic", None) for key in keys]
                row.extend(semantic_values)
                row.append(statistics.mean([v for v in semantic_values if v is not None]))
                
                # 处理 eye_catch 列
                eye_catch_values = [final_result[model].get(str(key), {}).get("eye_catch", None) for key in keys]
                row.extend(eye_catch_values)
                row.append(statistics.mean([v for v in eye_catch_values if v is not None]))
                
                # 处理 polish 列
                polish_values = [final_result[model].get(str(key), {}).get("polish", None) for key in keys]
                row.extend(polish_values)
                row.append(statistics.mean([v for v in polish_values if v is not None]))
                
                # 处理 original 列
                original_values = [final_result[model].get(str(key), {}).get("original", None) for key in keys]
                row.extend(original_values)
                row.append(statistics.mean([v for v in original_values if v is not None]))
                
                # 处理 Failed 列
                failed_values = [final_result[model].get(str(key), {}).get("failed", None) for key in keys]
                row.extend(failed_values)
                row.append(statistics.mean([v for v in failed_values if v is not None]))

                writer.writerow(row)
                    

