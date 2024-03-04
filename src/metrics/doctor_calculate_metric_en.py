import os
import re
import pdb
import json
import argparse
import numpy as np
import spacy
from rouge_score import rouge_scorer
from distinct_utils import distinct_n_corpus_level
# from ..utils.openai_utils import split_chinese_medicalinfo_and_question

nlp = spacy.load("en_core_web_sm")
scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

def remove_punctuation(text):
    # 使用正则表达式去除所有标点符号
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation

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
    Distinct2 = []
    Average_Turn = []
    Average_Length = []
    for data in datas:
        if data["raw_data"]["answer_idx"] == data["diagnosis"]:
            Accuracy.append(1)
        else:
            Accuracy.append(0)
        
        if "history" in data.keys():
            required_patient_info = ""
            dialog = []
            for history in data["history"]:
                dialog.append(remove_punctuation(history["doctor"]).split(" "))
                # pdb.set_trace()
                Average_Length.append(len(history["doctor"].split(" ")))

                if "patient" in history.keys():
                    required_patient_info += history["patient"]
                    dialog.append(remove_punctuation(history["patient"]).split(" "))
                # parse_info = nlp(history["patient"])
                # parse_entity_list += [ent.text for ent in parse_info.ents]

            Distinct2.append(distinct_n_corpus_level(dialog, 1))
            patient_info = data["raw_data"]["question"].rsplit(".", 1)[0]

            score = scorer.score(patient_info, required_patient_info)
            Coverage_Rouge1.append(score['rouge1'].recall)

            parse_info = nlp(patient_info)
            reference_entity_list = [ent.text for ent in parse_info.ents]
            
            if len(reference_entity_list) > 0:
                k_match = 0
                for ent in reference_entity_list:
                    if ent in required_patient_info:
                        k_match += 1
                Knowledge_Recall.append(k_match / len(reference_entity_list))
            
            Average_Turn.append(len(data["history"]))
    
    Accuracy = round(np.average(Accuracy),4)*100
    Coverage_Rouge1 = round(np.average(Coverage_Rouge1),4)*100
    Knowledge_Recall = round(np.average(Knowledge_Recall),4)*100
    Average_Turn = round(np.average(Average_Turn),4)
    Average_Length = round(np.average(Average_Length),4)
    Distinct2 = round(np.average(Distinct2),4)

    file_name = file_path.split("/")[-1].replace(".json", "")
    patient_model, doctor_model, _ = file_name.split('_')
    print("%15s %15s %15s %15s %15s %15s %15s %15s" %(patient_model, doctor_model, Accuracy, Coverage_Rouge1, Knowledge_Recall, Distinct2, Average_Turn, Average_Length))  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-path", type=str, default="/home/cs/yangyuchen/yushengliao/Medical_LLM/data/multiturn_pipeline/patient_test_results")
    
    args = parser.parse_args()
    print("%15s %15s %15s %15s %15s %15s %15s %15s" %('Patient Model', 'Doctor Model', "Accuracy", "Coverage_Rouge1", "Knowledge_Recall", "Distinct2", "Average_Turn", "Average_Length")) 
    calculate_metric(args)