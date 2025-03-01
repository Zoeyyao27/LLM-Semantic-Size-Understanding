from __future__ import print_function
import os
import random
import copy
import argparse
from pathlib import Path
import math
import pickle
import json
import glob
import time
import torch

# Data & Computation
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.metrics import roc_curve
from tqdm import tqdm

import plotly.express as px
import plotly.io as pio


import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
import seaborn as sns
from probe_LLMs.utils import *
import pandas as pd
from probe_LLMs.LM_hf import LM_nnsight




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="01-ai/Yi-6B-Chat") #"meta-llama/Llama-2-13b-chat-hf" "01-ai/Yi-6B-Chat" ""
    parser.add_argument('--csv_data_path', type=str, default="data/abstract2concrete_extend")
    parser.add_argument('--representation_path', type=str, default="output_probe/representation")
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--output_dir', type=str, default="output_probe")
    parser.add_argument('--max_iter', type=int, default=2000,help='max iteration for logistic regression')
    parser.add_argument('--C', type=float, default=1,help='regularization strength')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--if_not_extend', action='store_true',help='use the original dataset')
    parser.add_argument('--abstract_only', action='store_true',help='only probe abstract words')
    parser.add_argument('--concrete_only', action='store_true',help='only probe concrete words')
    parser.add_argument('--probe_hidden', action='store_true',help='probe hidden states instead of attention heads')
    parser.add_argument('--if_multinomial', action='store_true',help='probe using multinomial logistic regression')
    args = parser.parse_args()

    if args.probe_hidden:
        raise NotImplementedError("The hidden states are not implemented yet")
    ##assert only one of the abstract_only and concrete_only is true
    assert not (args.abstract_only and args.concrete_only), "Only one of the abstract_only and concrete_only can be true"

    ##concate the output_dir with args
    if args.probe_hidden:
        args.output_dir=os.path.join(args.output_dir,"hidden",f"max_iter_{args.max_iter}_C_{args.C}_seed_{args.seed}")
    else:
        if args.if_multinomial:
            args.output_dir=os.path.join(args.output_dir,"multinomial","attention",f"max_iter_{args.max_iter}_C_{args.C}_seed_{args.seed}")
        else:
            args.output_dir=os.path.join(args.output_dir,"attention",f"max_iter_{args.max_iter}_C_{args.C}_seed_{args.seed}")
    
    if args.abstract_only:
        args.output_dir=os.path.join(args.output_dir,f"abstract_only")
    if args.concrete_only:
        args.output_dir=os.path.join(args.output_dir,"concrete_only")

    if not args.abstract_only and not args.concrete_only:
        args.output_dir=os.path.join(args.output_dir,"all")

    ##if use extend data, then change the data path
    if args.if_not_extend:
        args.csv_data_path=args.csv_data_path.replace("_extend","")
    
    model_name=args.model_name.split("/")[-1]
    ensure_dir(args.representation_path,model_name,rewrite=False)

    args.representation_path=os.path.join(args.representation_path,model_name)
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

def plot_heatmap(ht, name, save_path=None):
    # Increase global font size for all text elements
    plt.rcParams.update({'font.size': 22})

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 8))


    #calculate the average for each row
    row_averages = np.mean(ht, axis=1)
    for i, avg in enumerate(row_averages):
        ax.text(ht.shape[1] + 0.5, i, f'{avg:.2f}', va='center', ha='left',fontsize=12)



    # 获取热力图的位置
    pos = ax.get_position()

    # 调整 colorbar 的位置和大小
    cbar_ax = fig.add_axes([pos.x1 + 0.05, pos.y0, 0.02, pos.height])

    #调整热力图的位置
    ax.set_position([pos.x0-0.05, pos.y0, pos.width * 0.9, pos.height])


    # Create a heatmap using seaborn
    sns.heatmap(ht, ax=ax,cbar_ax=cbar_ax, cmap='Greens', vmin=0.5, vmax=1, cbar_kws={'drawedges': False}, square=True)




    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(2)  # Set colorbar outline width

    # Set the ticks for x and y axes with specified interval
    ax.set_xticks(np.arange(0.5, ht.shape[1], 5))
    ax.set_yticks(np.arange(0.5, ht.shape[0], 5))

    # Set the tick labels for x and y axes with specified interval and keep x-axis labels horizontal
    ax.set_xticklabels(np.arange(0, ht.shape[1], 5), rotation=0)
    ax.set_yticklabels(np.arange(0, ht.shape[0], 5))

    # Set axis labels and title with increased padding and font size
    ax.set_xlabel('Head', fontsize=24, labelpad=20)
    ax.set_ylabel('Layer', fontsize=24, labelpad=20)
    # ax.set_title(name, fontsize=28)

    # Reinstate axis lines with specified linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_linewidth(2)

    # Optionally save the figure as a PDF with vectorized content
    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')

    # Clear the current figure's memory to prevent resource leaks
    plt.close(fig)
    

def probe_single_case(X_train, y_train, X_val, y_val, seed=0, verbose=False,max_iter=2000,C=3,if_multinomial=False):
    if if_multinomial:
        clf = LogisticRegression(random_state=seed, multi_class='multinomial', solver='lbfgs',max_iter=max_iter, C=C).fit(X_train, y_train)
        # assert False
    else:
        clf = LogisticRegression(random_state=seed, max_iter=max_iter, C=C).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    train_acc = accuracy_score(y_train, y_pred)

    y_val_proba = clf.predict_proba(X_val)[:, 1]  # Probability estimates for the positive class
    roc_auc = roc_auc_score(y_val, y_val_proba)
    logloss = log_loss(y_val, y_val_proba)
    
    if verbose:
        print("Confusion Matrix (Validation Set):")
        print(confusion_matrix(y_val, y_val_pred))
        # Classification Report
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, y_val_pred))
        # ROC-AUC Score
        print("\nROC-AUC Score (Validation Set):", roc_auc)
        print("\nLog-Loss (Validation Set):", logloss)
    return train_acc, val_acc, roc_auc, logloss, clf


def save_condition(args, words,output_path,temperature=0):

    llm = LM_nnsight(model_path=args.model_name, temperature=temperature, cahce_dir=args.cache_dir)
      
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    #for idx, word in tqdm(enumerate(words)):
    for idx, word in tqdm(enumerate(words), total=len(words), dynamic_ncols=True, leave=True):
        state_h, state_a = llm.get_all_states(prompt=word)
        # Extract last token
        state_h = state_h[:,-1]
        state_a = state_a[:,-1]

        path_h = os.path.join(output_path, f'reps_{word}_hidden.npy')
        path_a = os.path.join(output_path, f'reps_{word}_attention.npy')
        np.save(path_h, state_h)
        np.save(path_a, state_a)
 

def load_data(data_path,representation_path,args):
    ###if there is no representation file under the representation path,
    ###then we read the csv file and create the representation file

    ####read big csv file
    data_path=os.path.join(data_path, f"abstract2concrete_size_vary.csv")
    data = pd.read_csv(data_path, skiprows=1,encoding='utf-8')

    ##clean data
    data = data.replace('\xa0', ' ', regex=True)
    ##replace nan with empty string
    data = data.fillna('')

    # 遍历每一行
    final_word=[]
    final_word_label=[] #big  word has label 1 and small word has label 0
    for index, row in data.iterrows():
        row_list=row.tolist()
        ####abstract word
        if not args.concrete_only:
            abstract_big=row_list[1]
            abstract_small=row_list[4]

        ###concrete word
        if not args.abstract_only:
            concrete_big=row_list[6]
            concrete_small=row_list[8]

        ## word label
        big_label=1
        small_label=0

        ####store the data

        if not args.concrete_only:
            if abstract_big not in final_word:
                final_word.append(abstract_big)
                final_word_label.append(big_label)

            if abstract_small not in final_word:
                final_word.append(abstract_small)
                final_word_label.append(small_label)
        if not args.abstract_only:
            if concrete_big not in final_word:
                final_word.append(concrete_big)
                final_word_label.append(big_label)
            
            if concrete_small not in final_word:
                final_word.append(concrete_small)
                final_word_label.append(small_label)

    ####################

    assert len(final_word)==len(set(final_word)), "There are duplicate data!!"
    ##print the first 10 data 
    print("*************The first 10 data*************")
    print(final_word[:10])
    print(final_word_label[:10])

    if not os.listdir(representation_path):
        save_condition(args, final_word,representation_path,temperature=0)
    
    ##read the representation file
    feats_all=[]
    if args.probe_hidden:
        for word in final_word:
            path_h = os.path.join(representation_path, f'reps_{word}_hidden.npy')
            state_h = np.load(path_h)
            feats_all.append(state_h)
        feats_all = np.array(feats_all)
        labels_all = np.array(final_word_label)
    else:
        for word in final_word:
            #path_h = os.path.join(representation_path, f'reps_{word}_hidden.npy')
            path_a = os.path.join(representation_path, f'reps_{word}_attention.npy')
            #state_h = np.load(path_h)
            state_a = np.load(path_a)
            feats_all.append(state_a)

        feats_all = np.array(feats_all)
        labels_all = np.array(final_word_label)

    return feats_all, labels_all
    
    




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def probe_all(all_X, all_y, test_size=0.3, seed=0,max_iter=2000,C=3,if_multinomial=False):
    data_ids = np.arange(len(all_X))
    all_X_train, all_X_val, y_train, y_val, _, _ = train_test_split(all_X, all_y, data_ids, test_size=test_size, random_state=seed)

    num_layers, num_heads, head_dims = all_X_train.shape[1:]
    train_acc_all = np.zeros([num_layers, num_heads])
    val_acc_all = np.zeros([num_layers, num_heads])
    roc_auc_all = np.zeros([num_layers, num_heads])
    logloss_all = np.zeros([num_layers, num_heads])
    coefs_all = np.zeros([num_layers, num_heads, head_dims])
    CoMs_all = np.zeros([num_layers, num_heads, head_dims])
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            # print(layer, head)
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
            train_acc_all[layer][head], val_acc_all[layer][head], roc_auc_all[layer][head], logloss_all[layer][head], clf = probe_single_case(X_train, y_train, X_val, y_val, seed, max_iter=max_iter,C=C,if_multinomial=if_multinomial)          
            coefs_all[layer][head] = clf.coef_[0]
            # calculate mean 
            true_mass_mean = np.mean(X_train[y_train], axis=0)
            false_mass_mean = np.mean(X_train[y_train==False], axis=0)
            CoM_false2true = true_mass_mean - false_mass_mean
            CoMs_all[layer][head] = CoM_false2true
    return train_acc_all, val_acc_all, roc_auc_all, logloss_all, coefs_all, CoMs_all
    
if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)


    all_X, all_y = load_data(args.csv_data_path,args.representation_path,args)
    #assert False, "stop here"
    train_acc_all, val_acc_all, roc_auc_all, logloss_all, coefs_all, CoMs_all = probe_all(all_X, all_y, test_size=0.3, seed=args.seed,max_iter=args.max_iter,C=args.C,if_multinomial=args.if_multinomial)
    ##calculate the mean of the train_acc_all, val_acc_all, roc_auc_all across heads
    mean_train_acc_layers=np.mean(train_acc_all,axis=1)
    mean_val_acc_layers=np.mean(val_acc_all,axis=1)
    mean_roc_auc_layers=np.mean(roc_auc_all,axis=1)

    ##calculate the mean of the train_acc_all, val_acc_all, roc_auc_all across layers
    mean_train_acc_heads=np.mean(train_acc_all,axis=0)
    mean_val_acc_heads=np.mean(val_acc_all,axis=0)
    mean_roc_auc_heads=np.mean(roc_auc_all,axis=0)

    ##calculate the mean of the train_acc_all, val_acc_all, roc_auc_all across layers and heads
    mean_train_acc_all=np.mean(train_acc_all)
    mean_val_acc_all=np.mean(val_acc_all)
    mean_roc_auc_all=np.mean(roc_auc_all)

    


    # Save the results.
    model_name=args.model_name.split("/")[-1]
    


    ##save pdf 
    output_pdf_path=os.path.join(args.output_dir,"pdf")
    ensure_dir(output_pdf_path,model_name)

    plot_heatmap(train_acc_all, "Probe Train Acc.", save_path=os.path.join(output_pdf_path,model_name,"train_acc"))
    plot_heatmap(val_acc_all, "Probe Val Acc.", save_path=os.path.join(output_pdf_path,model_name,"val_acc"))
    plot_heatmap(roc_auc_all, "ROC AUC Val", save_path=os.path.join(output_pdf_path,model_name,"val_auc"))

    ##save npy
    output_npy_path=os.path.join(args.output_dir,"npy")
    ensure_dir(output_npy_path,model_name)

    np.save(os.path.join(output_npy_path,model_name,"val_acc.npy"), val_acc_all)
    np.save(os.path.join(output_npy_path,model_name,"coef.npy"), coefs_all)

    ##save all mean score into csv
    output_csv_path=os.path.join(args.output_dir,"csv")
    ensure_dir(output_csv_path,model_name)

    # 创建一个字典保存层的均值
    data_layers = {
        'Mean_Train_Acc_Layers': mean_train_acc_layers,
        'Mean_Val_Acc_Layers': mean_val_acc_layers,
        'Mean_Roc_Auc_Layers': mean_roc_auc_layers,
    }

    # 创建一个字典保存头的均值
    data_heads = {
        'Mean_Train_Acc_Heads': mean_train_acc_heads,
        'Mean_Val_Acc_Heads': mean_val_acc_heads,
        'Mean_Roc_Auc_Heads': mean_roc_auc_heads,
    }

    # 创建一个字典保存所有均值
    data_all = {
        'Mean_Train_Acc_All': [mean_train_acc_all],
        'Mean_Val_Acc_All': [mean_val_acc_all],
        'Mean_Roc_Auc_All': [mean_roc_auc_all],
    }

    # 将数据转换为DataFrame
    df_layers = pd.DataFrame(data_layers)
    df_heads = pd.DataFrame(data_heads)
    df_all = pd.DataFrame(data_all)

    # 将DataFrame保存为CSV文件
    df_layers.to_csv(os.path.join(output_csv_path,model_name,"mean_acc_results_layers.csv"), index=False)
    df_heads.to_csv(os.path.join(output_csv_path,model_name,"mean_acc_results_heads.csv"), index=False)
    df_all.to_csv(os.path.join(output_csv_path,model_name,"mean_acc_results_all.csv"), index=False)

  

    