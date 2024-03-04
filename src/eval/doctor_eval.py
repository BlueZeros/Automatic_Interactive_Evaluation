import os
import re
import pdb
import json
import argparse
import numpy as np
import pandas as pd
from Levenshtein import ratio
from progress.bar import Bar
from rouge_chinese import Rouge
from distinct_utils import distinct_n_sentence_level
import jieba
import scipy.stats as stats
from rouge_score import rouge_scorer

RESULT={"MODEL": [], "DIAGNOSIS": [], "COVERAGE": [], "INQUIRY_ACC": [], "INQUIRY_SPECIFIC": [], "INQUIRY_LOGIC": [], "ADVICE_ACC": [], "ADVICE_SPECIFIC": [], "DISTINCT": [], "AVG_TURN": [], "AVG_LEN": []}

def hisotry2str(historys, first_key="doctor", second_key="patient"):
    history_str = ""
    for history in historys:
        history_str += f"{history[first_key]}"
        if second_key in history.keys():
            history_str += f"{history[second_key]}\n"
    
    return history_str

def average(score_list):
    if score_list is None or score_list == []:
        return 0
    return sum(score_list) / len(score_list)

def sd(score_list):
    if score_list is None or score_list == []:
        return 0
    return np.std(score_list, ddof=1) / (len(score_list) ** 0.5)

def confidence_margin(score_list, p=0.95):
    if score_list is None or score_list == []:
        return 0
    
    std = np.std(score_list, ddof=1) # ddof=1 用于得到样本标准差
    n = len(score_list)
    z = stats.norm.ppf(1 - ((1 - p)/2)) # 使用0.975因为是双尾测试
    # 计算置信区间
    margin_of_error = z * (std / np.sqrt(n))
    
    return margin_of_error

def contains_chinese(s):
    return re.search('[\u4e00-\u9fff]', s) is not None

def get_rouge_score(args, hyposis, reference):
    if args.mode == "ninth":
        rouge_score = scorer.get_scores(' '.join(jieba.cut(hyposis)) , ' '.join(jieba.cut(reference)))[0]["rouge-1"]["r"]
    else:
        rouge_score = scorer.score(reference, hyposis)['rouge1'].recall

    return rouge_score

def get_lev_distance(args, hyposis, reference):
    if args.mode == "ninth":
        lev_distance = ratio(' '.join(jieba.cut(hyposis)), ' '.join(jieba.cut(reference)))
    else:
        lev_distance = ratio(hyposis, reference)
    return lev_distance

def get_distinct_score(args, history):
    # print(re.sub(r'[^\w\s]', '', ' '.join(jieba.cut(hisotry2str(history)))).split())
    if args.mode == "ninth":
        distinct_score = distinct_n_sentence_level(re.sub(r'[^\w\s]', '', ' '.join(jieba.cut(hisotry2str(history)))).split(), 2)
    else:
        distinct_score = distinct_n_sentence_level(re.sub(r'[^\w\s]', '', hisotry2str(history)).split(), 2)

    return distinct_score

def eval(args, file_name):
    with open(os.path.join(args.folder_path, file_name), "r") as f:
        datas = json.load(f)
    
    model_name = file_name.rsplit(".", 1)[0]
    # pattern = re.compile(r"gpt4-(.+?)-gpt4-gpt4")
    # model_name = pattern.search(model_name).group(1)

    dig = []
    cov = []
    inquiry_acc = []
    advice_acc = []
    logic = []
    inquiry_spec = []
    advice_spec = []
    distinct = []
    avg_turn = []
    avg_len = []

    for data in datas:
        ## diagnosis
        # dig.append(data["raw_data"]["answer_idx"] == data["diagnosis"])
        dig.append(list(data["turn_diagnosis"].values())[-1] == data["raw_data"]["answer_idx"])
        
        ## coverage rate
        patient_info_reference = data["raw_data"]["question"]
        require_patient_info = ""
        for turn in data["history"]:
            if turn["state"] in ["O", "A-A-A", "B-A-A"]:
                if "patient" not in turn:
                    turn["patient"] = ""
                require_patient_info += turn["patient"]
        
        coverage_score = get_rouge_score(args, require_patient_info, patient_info_reference)
        cov.append(coverage_score)

        # inquiry accuracy
        inquiry_success = [1 for turn in data["history"] if turn["state"] in ["A-A-A"]]
        advice_success = [1 for turn in data["history"] if turn["state"] in ["B-A-A"]]
        inquiry_specific_total = [1 for turn in data["history"] if turn["state"] in ["A-A-A", "A-A-B"]]
        advice_specific_total = [1 for turn in data["history"] if turn["state"] in ["B-A-A", "B-A-B"]]
        inquiry_total = [1 for turn in data["history"] if turn["state"] in ["A-A-A", "A-A-B", "A-B"]]
        advice_total = [1 for turn in data["history"] if turn["state"] in ["B-A-A", "B-A-B", "B-B"]]
        inquiry_acc.append(sum(inquiry_success) / (sum(inquiry_total) + 1e-9))
        advice_acc.append(sum(advice_success) / (sum(advice_total) + 1e-9))
        inquiry_spec.append(sum(inquiry_specific_total) / (sum(inquiry_total) + 1e-9))
        advice_spec.append(sum(advice_specific_total) / (sum(advice_total) + 1e-9))

        # consultation logic
        logic_score = get_lev_distance(args, require_patient_info, patient_info_reference)
        logic.append(logic_score)

        ## distinct-2
        dis_score = get_distinct_score(args, data["history"])
        distinct.append(dis_score)

        ## average turn
        avg_turn.append(len(data["history"]))
        avg_len += [len(turn["doctor"]) if contains_chinese(turn["doctor"]) else len(turn["doctor"].split(" ")) for turn in data["history"]]
        # if args.mode == "ninth":
        #     avg_len += [len(turn["doctor"]) for turn in data["history"]]
        # elif args.mode == "medqa":
        #     avg_len += [len(turn["doctor"].split(" ")) for turn in data["history"]]
        # else:
        #     raise NotImplementedError

        result = {key:0 for key in RESULT.keys()}
        result["MODEL"] = (model_name)
        result["DIAGNOSIS"] = (data["raw_data"]["answer_idx"] == data["diagnosis"])
        result["COVERAGE"] = ((coverage_score) * 100)
        result["INQUIRY_ACC"] = ((sum(inquiry_success) / (sum(inquiry_specific_total) + 1e-9)) * 100)
        result["INQUIRY_LOGIC"] = ((logic_score) * 100)
        result["INQUIRY_SPECIFIC"] = ((sum(inquiry_specific_total) / (sum(inquiry_total) + 1e-9)) * 100)
        result["ADVICE_ACC"] = ((sum(advice_success) / (sum(advice_specific_total) + 1e-9)) * 100)
        result["ADVICE_SPECIFIC"] = ((sum(advice_specific_total) / (sum(advice_total) + 1e-9)) * 100)
        result["DISTINCT"] = ((dis_score) * 100)

        data["eval_results"] = result
    
    with open(os.path.join(args.folder_path, file_name), "w") as f:
        json.dump(datas, f, indent=4, ensure_ascii=False)
    
    # mean = np.mean(numbers)
    # std = np.std(numbers, ddof=1) # ddof=1 用于得到样本标准差
    RESULT["MODEL"].append(model_name)
    RESULT["DIAGNOSIS"].append(f"{average(dig) * 100 :.2f}$\pm${sd(dig) * 100 :.2f}")
    RESULT["COVERAGE"].append(f"{average(cov) * 100 :.2f}$\pm${sd(cov) * 100 :.2f}")
    RESULT["INQUIRY_ACC"].append(f"{average(inquiry_acc) * 100 :.2f}$\pm${sd(inquiry_acc) * 100 :.2f}")
    RESULT["INQUIRY_LOGIC"].append(f"{average(logic) * 100 :.2f}$\pm${sd(logic) * 100 :.2f}")
    RESULT["INQUIRY_SPECIFIC"].append(f"{average(inquiry_spec) * 100 :.2f}$\pm${sd(inquiry_spec) * 100 :.2f}")
    RESULT["ADVICE_ACC"].append(f"{average(advice_acc) * 100 :.2f}$\pm${sd(advice_acc) * 100 :.2f}")
    RESULT["ADVICE_SPECIFIC"].append(f"{average(advice_spec) * 100 :.2f}$\pm${sd(advice_spec) * 100 :.2f}")
    RESULT["DISTINCT"].append(f"{average(distinct) * 100 :.2f}$\pm${sd(distinct) * 100 :.2f}")
    RESULT["AVG_TURN"].append(f"{average(avg_turn) :.2f}$\pm${sd(avg_turn) :.2f}")
    RESULT["AVG_LEN"].append(f"{average(avg_len) :.2f}$\pm${sd(avg_len) :.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ninth", "medqa"], required=True)
    parser.add_argument("--folder-path", type=str, default="/home/cs/yangyuchen/yushengliao/Medical_LLM/data/multiturn_pipeline/patient_test_results")
    args = parser.parse_args()
    
    files = os.listdir(args.folder_path)
    files = [file for file in files if re.search(r'\.json' , file)]
    files.sort() 

    if args.mode == "ninth":
        scorer = Rouge()
    elif args.mode == "medqa":
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    bar = Bar(f'EVALUATING', max=len(files), suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    for file in files:
        if "abc" not in file:
            print(file)
            eval(args, file)
            bar.next()
    
    df = pd.DataFrame(RESULT).round(2)
    
    print("\n")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.folder_path, 'result.csv'), index=False)



