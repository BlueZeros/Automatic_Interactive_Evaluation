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
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

openai.api_key = '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'

def patient_generate(data):
    # lock.acquire()
    # global count, seed_tasks
    # count += 1
    # data = data[0]
    # lock.release()
    data = data[0]
    if len(data["history"]) <= args.max_turn and ("?" in data["history"][-1]["doctor"] or "？" in data["history"][-1]["doctor"]):
        # pdb.set_trace()
        conv = get_patient_template(args.patient_conv_id).copy()
        conv.system_prompt_init(get_patient_prompt(args.patient_prompt_id))
        
        conv.init_history(data["history"], first_key="doctor", second_key="patient")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(patient_info= split_chinese_medicalinfo_and_question(data["question"])[0])

        # pdb.set_trace()
        message = [{"role": "assistant", "content": prompt}]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.0,
            max_tokens=120,
            stop= ['医生:','患者:']
            )  
        
        outputs = completion.choices[0].message["content"]
        data["history"][-1]["patient"] = outputs

    # seed_tasks.append(data)
    bar.next()

    # lock.acquire()
    # if len(seed_tasks) % 20 == 0: 
    #     np.save(f"{args.output_file_name}.npy", seed_tasks)
    # elif len(seed_tasks) % 10 == 0: 
    #     np.save(f"{args.output_file_name}_temp.npy", seed_tasks)
    # lock.release()

def doctor_generate(data):
    # lock.acquire()
    # global count, seed_tasks
    # count += 1
    # data = data[0]
    # lock.release()
    data = data[0]

    if len(data["history"]) <= 10 and (len(data["history"]) == 0 or "patient" in data["history"][-1].keys()):
        conv = get_doctor_template(args.doctor_conv_id).copy()
        conv.system_prompt_init(get_doctor_prompt(args.doctor_prompt_id))

        conv.init_history(data["history"], first_key="doctor", second_key="patient")
        conv.append_message(conv.roles[0], None)
        prompt = conv.get_prompt()

        # pdb.set_trace()
        message = [{"role": "assistant", "content": prompt}]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.0,
            max_tokens=120,
            stop= ['医生:','患者:']
            )  
        
        outputs = completion.choices[0].message["content"]
        data["history"].append({"doctor": outputs})

    # seed_tasks.append(data)
    bar.next()

    # lock.acquire()
    # if len(seed_tasks) % 20 == 0: 
    #     np.save(f"{args.output_file_name}.npy", seed_tasks)
    # elif len(seed_tasks) % 10 == 0: 
    #     np.save(f"{args.output_file_name}_temp.npy", seed_tasks)
    # lock.release()

parser = argparse.ArgumentParser()
parser.add_argument("--input-file-name", type=str, required=True)
parser.add_argument("--output-file-name", type=str, required=True)
parser.add_argument("--patient-conv-id", type=str, default="chatgpt_zh")
parser.add_argument("--patient-prompt-id", type=str, default="base_v1_zh")
parser.add_argument("--few-shot", type=int, default=0)
parser.add_argument("--doctor-conv-id", type=str, default="chatgpt_zh")
parser.add_argument("--doctor-prompt-id", type=str, default="base_v1_zh")
parser.add_argument("--max-turn", type=int, default=10)
args = parser.parse_args()

args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
args.output_file_name = args.output_file_name.rsplit(".", 1)[0]
# debug

lock=Lock()
count=0

seed_tasks, seed_idx = data_initialization(args)

if seed_tasks == []:
    with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
        origin_questions = json.load(f)
        for q in origin_questions:
            if q["id"] not in seed_idx:
                seed_tasks.append(q)

# seed_tasks = seed_tasks[:1]
total_count = len(seed_tasks)

for i in range(args.max_turn):
    bar = Bar(f'Processing Turn [{i}] Doctor', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    pool = ThreadPool(processes=10)
    res = pool.starmap(doctor_generate, [[i] for i in zip(seed_tasks)])
    pool.close()
    pool.join()
    bar.finish()

    bar = Bar(f'Processing Turn [{i}] Patient', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
    pool = ThreadPool(processes=10)
    res = pool.starmap(patient_generate, [[i] for i in zip(seed_tasks)])
    pool.close()
    pool.join()
    bar.finish()

    with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
        json.dump(seed_tasks, f, indent=4, ensure_ascii=False)