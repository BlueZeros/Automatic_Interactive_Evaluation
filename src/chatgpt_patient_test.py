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
os.environ["http_proxy"] = "http://127.0.0.1:51251"
os.environ["https_proxy"] = "http://127.0.0.1:51251"

openai.api_key = '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'

def data_initialization():
    if os.path.exists(f"{args.output_file_name}.npy") and os.path.exists(f"{args.output_file_name}_temp.npy"):
        seed_tasks = np.load(f"{args.output_file_name}.npy", allow_pickle=True).tolist()
        seed_tasks_temp = np.load(f"{args.output_file_name}_temp.npy", allow_pickle=True).tolist()
        if len(seed_tasks) > len(seed_tasks_temp):
            seed_idx = [s["id"] for s in seed_tasks]
        else:
            seed_idx = [s["id"] for s in seed_tasks_temp]
    elif os.path.exists(f"{args.output_file_name}.npy"):
        seed_tasks = np.load(f"{args.output_file_name}.npy", allow_pickle=True).tolist()
        seed_idx = [s["id"] for s in seed_tasks]
    elif os.path.exists(f"{args.output_file_name}_temp.npy"):
        seed_tasks = np.load(f"{args.output_file_name}_temp.npy", allow_pickle=True).tolist()
        seed_idx = [s["id"] for s in seed_tasks]
    elif os.path.exists(f"{args.output_file_name}.json"):
        with open(f"{args.output_file_name}.json", "r", encoding="utf-8") as f:
            seed_tasks = json.load(f)
        # seed_tasks = np.load(f"{args.output_file_name}.npy", allow_pickle=True).tolist()
        seed_idx = [s["id"] for s in seed_tasks]
    else:
        seed_tasks = []
        seed_idx = []
    
    return seed_tasks, seed_idx

def generate(data):
    lock.acquire()
    global count, seed_tasks, args
    count += 1
    data = data[0]
    lock.release()
    
    # pdb.set_trace()
    conv = get_patient_template(args.conv_id).copy()
    conv.system_prompt_init(get_patient_prompt(args.prompt_id))
    
    for task in ["pie_test", "cautious_test", "honest_test", "concentrate_test"]:
        output_list = []
        for question in data[task]["question"]: 
            conv.clean_message()
            # pdb.set_trace()

            conv.init_history(data["few_shot"], args.few_shot)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt(patient_info=data["raw_data"]["question"].rsplit(".", 1)[0])

            # pdb.set_trace()
            message = [{"role": "assistant", "content": prompt}]
            completion = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=message,
                temperature=0.0,
                max_tokens=100,
                stop= ['PATIENT','DOCTOR']
                )  
            
            outputs = completion.choices[0].message["content"]
            output_list.append(outputs)
        
        data[task]["output"] = output_list

    seed_tasks.append(data)
    # lock.acquire()
    bar.next()
    
    if len(seed_tasks) == 50 or len(seed_tasks) == 100 or len(seed_tasks) == 150: 
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(seed_tasks, f, indent=4, ensure_ascii=False)
    # elif len(seed_tasks) % 10 == 0: 
    #     np.save(f"{args.output_file_name}_temp.npy", seed_tasks)
    # lock.release()

parser = argparse.ArgumentParser()
# parser.add_argument("--save-path", type=str, default="/DB/data/yushengliao/Medical_LLM/Medical_Consultation_Evaluation/Medical_Consultation_Evaluation/results/patient_test")
parser.add_argument("--input-file-name", type=str, required=True)
parser.add_argument("--output-file-name", type=str, required=True)
parser.add_argument("--conv-id", type=str, default="chatgpt")
parser.add_argument("--prompt-id", type=str, default="base_v1_en_new")
parser.add_argument("--few-shot", type=int, default=0)
args = parser.parse_args()

args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

# debug

lock=Lock()
count=0

seed_tasks, seed_idx = data_initialization()

with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
    origin_questions = json.load(f)
    questions = []
    for q in origin_questions:
        if q["id"] not in seed_idx:
            questions.append(q)

# questions = questions[:5]
total_count = len(questions)

print('bar here')
bar = Bar('Processing', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')

# generate([questions[0]])
# pdb.set_trace()

print('building threads')
# for ques in questions:
#     generate(ques)
# pdb.set_trace()
pool = ThreadPool(processes=3)
# generate([questions[0]])
res = pool.starmap(generate, [[i] for i in zip(questions)])
pool.close()
pool.join()
bar.finish()

with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
    json.dump(seed_tasks, f, indent=4, ensure_ascii=False)

print(len(seed_tasks))
# os.remove(f"{args.output_file_name}.npy")
# os.remove(f"{args.output_file_name}_temp.npy")