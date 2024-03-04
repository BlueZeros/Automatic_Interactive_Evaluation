import json
import os
import openai
import pdb
import re
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
from random import sample
import argparse
import requests
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
from utils.general_utils import mkdir
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.openai_utils import data_initialization
from models import get_model

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def extract_chief_complain(args, model, data, bar):
    
    if "patient" in data["history"][0]:
        bar.next()
        return

    patient_info = data["raw_data"]["question"]
    if args.mode == "ninth":
        inputs = f"""
<患者信息>：{patient_info}

请从<患者信息>中提取原文作为患者的<主诉>，包含患者近期的主要感受以及前来就诊的直接原因。注意，不要透露患者的诊断结果，也不要透露过于具体详细的病症数值信息。<主诉>应该尽可能精简。

<主诉>："""
        
    elif args.mode == "medqa":
        inputs = f"""
<Patient Information>：{patient_info}
<Motivation>：{data["question"]}

Please extract the <Chief Complaint> from the provided <Patient Information>. <Chief Complaint> should include the patient's recent main feelings and the <Motivation> in first person. Be mindful not to disclose the patient's diagnosis or too specific and detailed symptom values. The <Chief Complaint> should be only one sentence.

<Chief Complaint>："""
        
    else:
        raise NotImplementedError

    # message = [{"role": "user", "content": inputs}]
    response = model.generate(inputs)
    
    # response = completion.choices[0].message.content
    data["history"][0]["patient"] = response

    if args.debug:
        print("===========================================================")
        print(f"[PROMPT]\n{inputs}\n\n[OUTPUT]\n{response}\n\n")
        model.log()
        pdb.set_trace()
    bar.next()

TASK_LIST = {
    "extract_chief_complain": extract_chief_complain,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['medqa', 'medicaleval', 'ninth'], required=True)
    parser.add_argument("--input-file-name", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)
    
    parser.add_argument("--model", choices=["chatgpt", "gpt4"], default="chatgpt")
    parser.add_argument("--tasks", choices=["extract_chief_complain"], default="extract_chief_complain")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
    args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

    mkdir(args.output_file_name)
    
    # data prepare process
    datas, _ = data_initialization(args)
    if datas == []:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            for q in origin_questions:
                datas.append(q)
    
    if args.debug:
        # datas = datas[:10]
        args.workers = 1
        # datas[0]["history"] = []
        
    total_count = len(datas)
    generate = TASK_LIST[args.tasks]
    model = get_model(args.model)

    bar = Bar(f'Processing Task {args.tasks}', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    pool = ThreadPool(processes=args.workers)
    pool.starmap(generate, [[args, model, data, bar] for data in datas])
    pool.close()
    pool.join()
    bar.finish()

    model.log()

    if not args.debug:
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)