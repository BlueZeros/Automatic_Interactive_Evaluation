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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
from utils.general_utils import mkdir
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.agent import Doctor_Agent_V3, Patient_Agent_V3, StateDetect_Agent_V4, Dignosis_Agent
from models import get_model, API_Model, XingHuo_Model, QianWen_Model, YiYan_Model
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def model_initialization(args):
    doctor_model = get_model(args.doctor_model, [])
    print(f"[Doctor Model] {type(doctor_model)}")

    return doctor_model

def generate(args, model, data, hl, bar):

    if "turn_diagnosis" not in data:
        data["turn_diagnosis"] = {}
    
    # the case with results and shorter history is skipped
    if hl in data["turn_diagnosis"] or str(hl) in data["turn_diagnosis"]: #or len(data["history"]) < hl:
        bar.next()
        return

    conv = get_doctor_template(args.mode, "gpt4").copy()
    conv.system_prompt_init("")
    conv.init_history(data["history"], first_key="doctor", second_key="patient", turn=hl, latest=False)
    diaglog = conv.get_prompt()
    
    if args.condition == "dialogue":
        prompt = f"\
**Conversation:**\n{diaglog}\n\
Please choose the correct answer for the following question according to the consultation conversation between the doctor and the patient. Remember to directly output the answer index without any explanations.\n\
**Question:** {data['question']}\n\
**Options:** {data['raw_data']['options']}\n\
**Answer:**("
    else:
        prompt = f"\
**Paitent Information:**\n{data['raw_data']['question']}\n\
Please choose the correct answer for the following question according to the patient information. Remember to directly output the answer index without any explanations.\n\
**Question:** {data['question']}\n\
**Options:** {data['raw_data']['options']}\n\
**Answer:**("
    
    if isinstance(data["raw_data"]["options"], dict):
        n = len(data["raw_data"]["options"].keys())
    elif isinstance(data["raw_data"]["options"], str):
        # 编译正则表达式
        pattern = re.compile(r'\([A-Z]\)')
        # 查找所有匹配项
        n = len(pattern.findall(data["raw_data"]["options"]))
    else:
        raise NotImplementedError
    
    loget_bias = model.get_logit_bias(state_num=n)
    if loget_bias is not None:
        outputs = model.multiple_choice_selection(prompt, loget_bias)
    else:
        outputs = model.generate(prompt)
    
    data["turn_diagnosis"][hl] = outputs

    if args.debug:
        print(f"==============  Diagnosis ===================")
        print(f"Prompt: {prompt}")
        print(f"Output: {outputs}")
        pdb.set_trace()

    bar.next()

def generate_forward(args, model, datas, hl, bar):
    if isinstance(model, API_Model):
        if isinstance(model, XingHuo_Model) or isinstance(model, QianWen_Model) or isinstance(model, YiYan_Model):
            pool = ThreadPool(processes=1)
        else:
            pool = ThreadPool(processes=args.workers)
            
        pool.starmap(generate, [[args, model, data, hl, bar] for data in datas])
        pool.close()
        pool.join()
    else:
        for data in datas:
            generate(args, model, data, hl, bar)
    
    model.log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['medqa', 'medicaleval', 'ninth'], required=True)
    parser.add_argument("--condition", choices=["dialogue", "patient_info"], required=True)
    parser.add_argument("--input-file-name", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)

    parser.add_argument("--doctor-model", type=str, default="chatgpt")
    parser.add_argument("--history-len", type=int, default=-1)
    parser.add_argument("--history-len-b", type=int, default=None)
    parser.add_argument("--history-len-e", type=int, default=None)

    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[Input File] {args.input_file_name}")
    print(f"[Output File] {args.output_file_name}")
    args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
    args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

    # mkdir
    mkdir(args.output_file_name)

    # data prepare process
    datas, _ = data_initialization(args)
    if datas == []:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            for q in origin_questions:
                if "history" not in q.keys():
                    q["history"] = []
                datas.append(q)
    
    # agent
    doctor_model = model_initialization(args)
    
    if args.debug:
        datas = datas[:1]
        args.workers = 1

    total_count = len(datas)

    if args.history_len_b is None or args.history_len_e is None or args.condition == "patient_info":
        args.history_len = 0
        args.history_len_b =  args.history_len
        args.history_len_e = args.history_len

    for hl in range(args.history_len_b, args.history_len_e+1):
        bar = Bar(f'Processing Diagnosis with History Len [{hl}]', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(args, doctor_model, datas, hl, bar)
        bar.finish()

        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)