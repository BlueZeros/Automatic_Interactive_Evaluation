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
from utils.agent import Patient_Agent_Test, Patient_Agent_Test_W_State_V2
from models import get_model, API_Model
from patient_test_generator import INSTRUCTIONS
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def agent_initialization(args):
    # patient agent
    patient_conv = get_patient_template(args.mode, args.patient_model).copy()
    patient_model = get_model(args.patient_model, patient_conv.stop_ids)
    if args.state_aware:
        patient_agent = Patient_Agent_Test_W_State_V2(args, patient_model, patient_conv)
    else:
        patient_agent = Patient_Agent_Test(args, patient_model, patient_conv)
    print(f"[Patient Conv] {type(patient_conv)}")
    print(f"[Patient Model] {type(patient_model)}")
    print(f"[Patient Agent] {type(patient_agent)}")

    return patient_agent

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

    parser.add_argument("--patient-prompt-id", type=str, default="base_v1_new")
    parser.add_argument("--patient-model", type=str, default="chatgpt")
    parser.add_argument("--state-aware", action="store_true", default=False)
    parser.add_argument("--golden_state", action="store_true", default=False)

    parser.add_argument("--question-type", type=str, default=None)
    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cover", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[Input File] {args.input_file_name}")
    print(f"[Output File] {args.output_file_name}")
    args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
    args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

    if "state_aware" in args.patient_model:
        args.patient_model = args.patient_model.replace("_state_aware", "")
        args.state_aware = True

    # mkdir
    mkdir(args.output_file_name)

    if args.question_type is None:
        question_type_list = list(INSTRUCTIONS.keys())
    else:
        question_type_list = args.question_type.split(",")

    # data prepare process
    datas, _ = data_initialization(args)
    datas.sort(key=lambda x:x["id"])
    if datas != [] and args.cover:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            origin_questions.sort(key=lambda x:x["id"])
            assert len(datas) == len(origin_questions)
            for (origin_data, data) in zip(origin_questions, datas):
                assert origin_data["id"] == data["id"]
                # pdb.set_trace()
                for hl in range(args.max_turn):
                    for qt in question_type_list:
                        data["patient_test"][str(hl)][qt] = origin_data["patient_test"][str(hl)][qt]
    elif datas != [] and not args.cover:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            origin_questions.sort(key=lambda x:x["id"])
            assert len(datas) == len(origin_questions)
            for (origin_data, data) in zip(origin_questions, datas):
                assert origin_data["id"] == data["id"]
                # pdb.set_trace()
                for hl in range(args.max_turn):
                    for qt in question_type_list:
                        if qt not in data["patient_test"][str(hl)].keys() or "prediction" not in data["patient_test"][str(hl)][qt].keys():
                            data["patient_test"][str(hl)][qt] = origin_data["patient_test"][str(hl)][qt]
    elif datas == []:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            for q in origin_questions:
                datas.append(q)
    
    # agent
    patient_agent = agent_initialization(args)

    # workers
    if args.patient_model in ["yiyan", "qianwen", "xinghuo"]:
        args.workers = 1
    
    if args.debug:
        datas = datas[:1]
        args.workers = 1
        # datas[0]["history"] = []
        
    total_count = len(datas)
    # pdb.set_trace()

    for hl in range(args.max_turn):
        for question_type in question_type_list:
            if args.state_aware:
                bar = Bar(f'Question Type: {question_type}, History Len: {hl}, State Detect I', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
                generate_forward(patient_agent.state_agent, datas, question_type, hl, "stageI", bar)
                bar.finish()

                bar = Bar(f'Question Type: {question_type}, History Len: {hl}, State Detect II', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
                generate_forward(patient_agent.state_agent, datas, question_type, hl, "stageII", bar)
                bar.finish()

                bar = Bar(f'Question Type: {question_type}, History Len: {hl}, State Detect III', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
                generate_forward(patient_agent.state_agent, datas, question_type, hl, "stageIII", bar)
                bar.finish()

            ## Patient generation
            bar = Bar(f'Question Type: {question_type}, History Len: {hl}', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
            generate_forward(patient_agent, datas, question_type, hl, bar)
            bar.finish()

            patient_agent.log()
            if not args.debug:
                with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(datas, f, indent=4, ensure_ascii=False)


    if not args.debug:
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)