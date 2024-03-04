import json
import os
import openai
import pdb
import re
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
from random import sample
import time
import argparse
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.patient_conversation import get_patient_template, get_patient_prompt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

openai.api_key = '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'

def generate(data, thread_id):
    lock.acquire()
    global count, seed_tasks
    count += 1

    lock.release()

    _, question = split_chinese_medicalinfo_and_question(data["question"])
    options = data["options"]

    conv = get_patient_template(args.patient_conv_id).copy()
    conv.system_prompt_init('{patient_info}')
    
    conv.init_history(data["history"], first_key="doctor", second_key="patient")
    patient_info = conv.get_prompt(patient_info='')
    
    # pdb.set_trace()
    prompt = f"Please choose the correct options for the following question according to the dialog about consultation. **Dialog:**{{{patient_info}}}**Question:** {{{question}}} {{{options}}} **Answer:**("

    logit_bias = {}
    for idx in range(len(options.keys())):
        logit_bias[idx+32] = 100

    message = [{"role": "assistant", "content": prompt}]
    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=message,
        logit_bias=logit_bias,
        temperature=0.0,
        max_tokens=1,
        )   
    response = completion.choices[0].message["content"]
    data["output"] = response
    seed_tasks.append(data)
    
    bar.next()

parser = argparse.ArgumentParser()
parser.add_argument("--input-file-name", type=str, required=True)
parser.add_argument("--output-file-name", type=str, required=True)
parser.add_argument("--patient-conv-id", type=str, default="chatgpt_zh")

args = parser.parse_args()

args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

lock=Lock()
count=0

seed_tasks, seed_idx = data_initialization(args)

with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
    origin_questions = json.load(f)
    questions = []
    for q in origin_questions:
        if q["id"] not in seed_idx:
            questions.append(q)

total_count = len(questions)
thread_id = [i%100 for i in range(len(questions))]

print('bar here')
bar = Bar('Processing', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')

print('building threads')
pool = ThreadPool(processes=10)
res = pool.starmap(generate, [[i, id] for (i, id) in zip(questions, thread_id)])
pool.close()
pool.join()
bar.finish()

with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
    json.dump(seed_tasks, f, indent=4, ensure_ascii=False)