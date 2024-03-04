import os
import re
import pdb
import json
import argparse
import numpy as np
import spacy
# from rouge_score import rouge_scorer
from rouge_chinese import Rouge
from Levenshtein import ratio
from distinct_utils import distinct_n_sentence_level
import jieba # you can use any other word cutting library
# from utils.openai_utils import split_chinese_medicalinfo_and_question
rouge = Rouge()
# scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

def hisotry2str(historys, first_key="doctor", second_key="patient"):
    history_str = ""
    for history in historys:
        history_str += f"{first_key}: {history['doctor']}\n"
        if second_key in history.keys():
            # assert history == historys[-1], f"{historys}\nCONVERSATION MISS ERROR!"
            history_str += f"{second_key}: {history['patient']}\n"
    
    return history_str

def calculate_metric(args):
    files = os.listdir(args.folder_path)
    files.sort() 
    pattern = r'\.json' 

    for file in files:
        # if file.endswith(r"\d+_multiple_6\.json"):
        if re.search(pattern, file):
            eval(os.path.join(args.folder_path, file))

def eval(file_path):
    with open(file_path, "r") as f:
        datas = json.load(f)
    
    Accuracy = []
    Coverage_Rouge1 = []
    Knowledge_Recall = []
    Consultation_Logic = []
    Distinct = []
    Average_Turn = []
    Average_Length = []
    for data in datas:
        if data["raw_data"]["answer_idx"] == data["diagnosis"]:
            Accuracy.append(1)
        else:
            Accuracy.append(0)

        if "history" in data.keys():
            required_patient_info = ""
            for history in data["history"][:-1]:
                if "patient" in history.keys():
                    required_patient_info += history["patient"]
                Average_Length.append(len(history["doctor"]))
                # parse_info = nlp(history["patient"])
                # parse_entity_list += [ent.text for ent in parse_info.ents]

            patient_info = data["raw_data"]["question"]
            if required_patient_info == "":
                required_patient_info = "None"
            rouge_score = rouge.get_scores(' '.join(jieba.cut(patient_info)) , ' '.join(jieba.cut(required_patient_info)))
            # pdb.set_trace()
            logic_score = ratio(re.sub(r'[^\w\s]', '', patient_info), re.sub(r'[^\w\s]', '', required_patient_info))
            # pdb.set_trace()
            distinct_score = distinct_n_sentence_level(re.sub(r'[^\w\s]', '', ' '.join(jieba.cut(hisotry2str(data["history"], "", "")))).split(), 2)

            Coverage_Rouge1.append(rouge_score[0]['rouge-1']['r'])
            Consultation_Logic.append(logic_score)
            Distinct.append(distinct_score)

            # reference_entity_list = [ent.text for ent in parse_info.ents]
            # pdb.set_trace()
            
            if len(data["entities"]) > 0:
                k_match = 0
                for ent in data["entities"]:
                    if ent in required_patient_info:
                        k_match += 1
                Knowledge_Recall.append(k_match / len(data["entities"]))
            
            Average_Turn.append(len(data["history"]))
    
    Accuracy = round(np.average(Accuracy),4)*100
    Coverage_Rouge1 = round(np.average(Coverage_Rouge1),4)*100
    Knowledge_Recall = round(np.average(Knowledge_Recall),4)*100
    Consultation_Logic = round(np.average(Consultation_Logic),4)*100
    Distinct = round(np.average(Distinct),4)*100
    Average_Turn = round(np.average(Average_Turn),4)
    Average_Length = round(np.average(Average_Length),4)

    file_name = file_path.split("/")[-1].replace(".json", "")
    print("%40s %.2f %.2f %.2f %.2f %.2f %.2f %.2f" %(file_name, Accuracy, Coverage_Rouge1, Knowledge_Recall, Consultation_Logic, Distinct, Average_Turn, Average_Length))  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/home/cs/yangyuchen/yushengliao/Medical_LLM/data/multiturn_pipeline/patient_test_results")
    
    args = parser.parse_args()
    print("%40s %15s %15s %15s %15s %15s %15s %15s" %('Model', "Accuracy", "Coverage_Rouge1", "Knowledge_Recall", "Consultation_Logic", "Distinct", "Average_Turn", "Average_Length")) 
    calculate_metric(args)