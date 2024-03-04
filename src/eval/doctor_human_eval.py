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

RESULT={"MODEL": [], "Inquiry": [], "Logic": [], "Patient": [], "Diagnosis": [], "Total": [], "Professional": [], "Effective": [], "Clear": [], "Understand": [], "Empathy": []}
INDEX = {"chatglm3": "ChatGLM", "internlm": "InterLM", "baichuan": "BaiChuan", "xinghuo": "XingHuo", "qianwen": "QianWen", "chatgpt": "ChatGPT", "gpt4": "GPT4"}

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

def eval(args):
    with open(os.path.join(args.folder_path, "patient.json"), "r") as f:
        patient_datas = json.load(f)
    
    with open(os.path.join(args.folder_path, "doctor.json"), "r") as f:
        doctor_datas = json.load(f)

    for model in INDEX:
    # print(model)
        RESULT["MODEL"].append(INDEX[model])
        for metric in [m for m in RESULT if m != "MODEL"]:
            temp_log = []
            for data in (doctor_datas + patient_datas):
                if "result" not in data or model not in [data["model1"]["model"], data["model2"]["model"]] or metric not in data["result"]:
                    continue
                
                if model == data["model1"]["model"]:

                    if "model1" in data["result"][metric]:
                        temp_log.append(1)
                    elif "model2" in data["result"][metric]:
                        temp_log.append(0)
                    elif "tie" in data["result"][metric]:
                        pass
                    else:
                        raise NotImplementedError
                    
                elif model == data["model2"]["model"]:
                    # print(data["result"], type(data["result"]))
                    if "model2" in data["result"][metric]:
                        temp_log.append(1)
                    elif "model1" in data["result"][metric]:
                        temp_log.append(0)
                    elif "tie" in data["result"][metric]:
                        pass
                    else:
                        raise NotImplementedError
                
                else:
                    raise NotImplementedError

            # average_success_rate = average(temp_log)
            # sd = sd(temp_log)
            RESULT[metric].append(f"{average(temp_log) * 100 :.2f}$\pm${sd(temp_log) * 100 :.2f}")
        
    
    # with open(os.path.join(args.folder_path, file_name), "w") as f:
    #     json.dump(datas, f, indent=4, ensure_ascii=False)
    
    # mean = np.mean(numbers)
    # std = np.std(numbers, ddof=1) # ddof=1 用于得到样本标准差
    # RESULT["MODEL"].append(model_name)
    # RESULT["DIAGNOSIS"].append(f"{average(dig) * 100 :.2f}$\pm${sd(dig) * 100 :.2f}")
    # RESULT["COVERAGE"].append(f"{average(cov) * 100 :.2f}$\pm${sd(cov) * 100 :.2f}")
    # RESULT["INQUIRY_ACC"].append(f"{average(inquiry_acc) * 100 :.2f}$\pm${sd(inquiry_acc) * 100 :.2f}")
    # RESULT["INQUIRY_LOGIC"].append(f"{average(logic) * 100 :.2f}$\pm${sd(logic) * 100 :.2f}")
    # RESULT["INQUIRY_SPECIFIC"].append(f"{average(inquiry_spec) * 100 :.2f}$\pm${sd(inquiry_spec) * 100 :.2f}")
    # RESULT["ADVICE_ACC"].append(f"{average(advice_acc) * 100 :.2f}$\pm${sd(advice_acc) * 100 :.2f}")
    # RESULT["ADVICE_SPECIFIC"].append(f"{average(advice_spec) * 100 :.2f}$\pm${sd(advice_spec) * 100 :.2f}")
    # RESULT["DISTINCT"].append(f"{average(distinct) * 100 :.2f}$\pm${sd(distinct) * 100 :.2f}")
    # RESULT["AVG_TURN"].append(f"{average(avg_turn) :.2f}$\pm${sd(avg_turn) :.2f}")
    # RESULT["AVG_LEN"].append(f"{average(avg_len) :.2f}$\pm${sd(avg_len) :.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/home/cs/yangyuchen/yushengliao/Medical_LLM/data/multiturn_pipeline/patient_test_results")
    args = parser.parse_args()

    # bar = Bar(f'EVALUATING', max=len(), suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    eval(args)
    
    df = pd.DataFrame(RESULT).round(2)
    
    print("\n")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.folder_path, 'result.csv'), index=False)



