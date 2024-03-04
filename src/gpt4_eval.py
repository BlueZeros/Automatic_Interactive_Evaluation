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
from utils.general_utils import mkdir, chunk_split
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.agent import EvalAgent
from models import get_model, API_Model

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def agent_initialization(args):
    # patient agent
    eval_conv = get_doctor_template(args.mode, "gpt4").copy()
    eval_model = get_model("gpt4", eval_conv.stop_ids)
    eval_agent = EvalAgent(args, eval_model, eval_conv)
    print(f"[Eval Conv] {type(eval_conv)}")
    print(f"[Eval Model] {type(eval_conv)}")
    print(f"[Eval Agent] {type(eval_agent)}")

    return eval_agent

def generate_forward(agent, datas, *kwargs):
    if isinstance(agent.model, API_Model):
        pool = ThreadPool(processes=args.workers)
        pool.starmap(agent.generate, [[data] + list(kwargs) for data in datas])
        pool.close()
        pool.join()
    else:
        for data in datas:
            agent.generate(data, *kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['medqa', 'medicaleval', 'ninth'], required=True)
    parser.add_argument("--input-file-name", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)

    parser.add_argument("--eval-type", choices=["doctor", "patient"], required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
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
            datas = json.load(f)
    
    chunked_datas, chunk_len = chunk_split(datas, args.chunk_size)

    # agent
    eval_agent = agent_initialization(args)

    # workers
    if args.debug:
        datas = datas[:1]
        args.workers = 1
        # datas[0]["history"] = []
    
    total_count = len(datas)
    # pdb.set_trace()

    for idx, chunk in enumerate(chunked_datas):
        bar = Bar(f'Chunk [{idx+1}/{chunk_len}]', max=len(chunk), suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        generate_forward(eval_agent, chunk, bar)
        bar.finish()

        eval_agent.log()
        if not args.debug:
            with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                json.dump(datas, f, indent=4, ensure_ascii=False)

        # if idx+1 >= 10:
        #     break

    if not args.debug:
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)