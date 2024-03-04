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

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

client = OpenAI(
    api_key="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"
)

INSTRUCTIONS = {
    ## 询问信息类--具体问题--信息在患者信息中
    "A-A-A": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行询问，患者回答医生的问题。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的问题只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生询问患者一个较为具体的问题，这个问题能够完全用<患者信息>中某一部分的原文回复。患者则会用<患者信息>种对应部分的原文进行回复。注意，医生的问题要和患者的回复对应，不要出现答非所问的情况。\n\
    4. 给出一个例子以供参考：\n\
        <例>\n\
            '患者信息': 女，20 岁，左侧鼻前庭肿物 1 年，增大半月”入院。 查体：左侧鼻底、鼻旁隆起，鼻外形正常，鼻中隔居中，双侧鼻腔黏膜光滑、充血，鼻窦区无压痛。辅检：2021.7.5（我院）颌面部CT 示左侧鼻前庭异常组织影，鼻前庭囊肿待排，建议增强检查。，2021-07-07 12:04：WBC 白细胞计数，10.2 10^9/L，↑；NEUT# 中性粒细胞绝对数，7.01 x10^9/L，↑；MONO# 单核细胞绝对数，0.70 x10^9/L，↑；BASO# 嗜碱性粒细胞绝对数，0.07 x10^9/L，↑；RDW-SD 红细胞体积分布-SD，38.20 fL，↓；PLT 血小板计数，355 x10^9/L，↑；2021-07-07 13:14：INR 国际标准化比值，0.71，↓；2021-07-07 13:16：CHE 胆碱酯酶，12.54 KU/L，↑；GLU 葡萄糖，6.5 mmol/L，↑；TG 甘油三酯，1.77 mmol/L，↑；ApoCⅡ 载脂蛋白 CⅡ，5.25 mg/dL，↑；ApoCⅢ 载脂蛋白 C-Ⅲ，12.06 mg/dL，↑；NEFA 游离脂肪酸，0.51 mmol/L，↑；2021-07-07 14:16：HbA1C 糖化血红蛋白，7.0 %，↑；，7.7 心电图：正常范围心电图；胸片：两肺未见明显活动性病变。\n\
            正例: \n\
                 [医生]：请问您鼻子的症状到现在已经多久了呢？最近有什么新的变化吗？\n\
                 [患者]：左侧鼻前庭肿物 1 年，增大半月。\n\
                 解释：这里患者的回复是'患者信息'中的原文，且医生的问题很具体，满足要求。\n\
            正例: \n\
                 [医生]：请问您的鼻前庭肿物有检查过么，如果有能告诉我查体结果是什么吗？\n\
                 [患者]：查体：左侧鼻底、鼻旁隆起，鼻外形正常，鼻中隔居中，双侧鼻腔黏膜光滑、充血，鼻窦区无压痛。\n\
                 解释：这里患者的回复是'患者信息'中的原文，且医生的问题很具体，满足要求。\n\
            负例: \n\
                [医生]：你有做过什么检查么，如果有能告诉我检查结果是什么吗？\n\
                [患者]：查体：左侧鼻底、鼻旁隆起，鼻外形正常，鼻中隔居中，双侧鼻腔黏膜光滑、充血，鼻窦区无压痛。\n\
                解释：这里医生询问时没有给出特定的检查项目，不够具体。不满足要求。\n\
            负例: \n\
                [医生]：您好，我注意到您的血糖水平有些升高，您有没有感觉到经常口渴、尿频或者是体重有无故的下降？\n\
                [患者]：是的，我确实有这些症状。我的葡萄糖水平是6.5 mmol/L，稍微高了一些。\n\
                解释：首先，患者的回复并不是'患者信息'中的原文，而且医生的问题‘您有没有感觉到经常口渴、尿频或者是体重有无故的下降？’在'患者信息'中完全没有提及，所以不满足要求。\n\n",

    ## 询问信息类--具体问题--信息不在患者信息中
    "A-A-B": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行询问，患者回答医生的问题。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的问题只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生的问题应该较为具体，如询问具体的症状类型或者检查类型，但是这个问题在<患者信息>中没有对应的答案，患者应该予以否定。\n",

    ## 询问信息类--宽泛问题
    "A-B": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行询问，患者回答医生的问题。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的问题只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生的问题应该较为宽泛，多为开放式问题，在问题中不提及具体的医疗名词，不要举例子，不要给出具体的症状部位感觉要求，问题越模糊越好，尽量让患者不能理解。\n\
    4. 患者的回复会要求医生询问得更加具体一点！而不会直接回答问题！\n",

    ## 医疗建议类--具体问题--信息在患者信息中
    "B-A-A": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行建议，患者回答医生。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的建议只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生建议患者去做<患者信息>中有相关检查结果的结果的具体检查项目，患者的回复应该直接复制<患者信息>中有关该检查建议的数值和结果的文本段。\n\
    4. 给出两个例子以供参考：\n\
        例1：如果<患者信息>中有患者的CT检查结果，那么医生应该建议患者去做CT检查，而患者应该回复该检查的相关结果。\n\
        例2：如果<患者信息>中只提到了”建议做MRI”，但没有具体的MRI检查结果，那么医生就应该如例1所示建议患者做别的<患者信息>中有提到结果的具体检查或治疗。\n",

    ## 医疗建议类--具体问题--信息不在患者信息中
    "B-A-B": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行建议，患者回答医生。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的建议只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生给出的建议应该较为具体，涉及具体的医疗名词，但<患者信息>中没有该建议及其结果的记录。患者应该回复没做过该建议，感谢会听从该建议。\n",

    ## 医疗建议类--宽泛问题--信息不在患者信息中
    "B-B": "请根据下述要求生成一轮的医生患者的对话。医生对患者进行建议，患者回答医生。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的建议只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生给出的建议应该较为宽泛，如建议去医院治疗做检查等较为模糊不具体的建议，不涉及具体的医疗名词。患者的回复会要求医生询问得更加具体一点。\n",

    ## 物理动作类
    "C": "请根据下述要求生成一轮的医生患者的对话。医生先，患者后。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的问题只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 为了检查患者的问题，医生要求患者做出某些动作以配合医生的检查。患者的回复应该提醒医生这是网络问诊，并表示不能完成这个动作，患者的回复应该与<患者信息>的事实一致。\n",

    ## 其他话题类
    "D": "请根据下述要求生成一轮的医生患者的对话。医生先，患者后。要求：\n\
    1. 请直接生成且仅生成一轮医生和患者的对话。医生的问题只涉及要求3中的内容，不要过于复杂。\n\
    2. 该轮医生和患者的对话承接<对话历史>，在逻辑上通顺。注意，如果没有<对话历史>，那么该轮对话是医生和患者的第一轮对话。\n\
    3. 医生问的问题或者说的话严重偏离了医疗问诊的主题，而是询问患者的其他与医疗领域无关的事宜。注意偏离的程度应该较大，询问饮食习惯和生活习惯也属于医疗相关主题。患者的回复应该要求医生回到问诊的情景下。\n",
}
    

def generate_qa(args, data, history_len, question_type, bar):
    conv = get_patient_template(args.mode, "chatgpt").copy()
    conv.system_prompt_init("{patient_info}")
    # if history_len != 0:
    conv.init_history(data["history"][:history_len], first_key="doctor", second_key="patient")
    history_src = conv.get_prompt(patient_info="")
    patient_info = data["raw_data"]["question"]
    instructions = INSTRUCTIONS[question_type]

    if question_type in data["patient_test"][str(history_len)].keys() and not args.cover:
        bar.next()
        return

    inputs = ""
    if history_len != 0:
        inputs += f"<对话历史>:\n{history_src}\n\n"

    if question_type not in ["A-B", "B-B", "C", "D"]:
        inputs += f"<患者信息>:\n{patient_info}\n\n"
    
    inputs += f"{instructions}"

    message = [{"role": "user", "content": inputs}]
    
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=message,
        temperature=0.0,
        max_tokens=500,
        )
    
    response = completion.choices[0].message.content
    # print(response)
    # pdb.set_trace()

    data["patient_test"][str(history_len)][question_type] = response

    if args.debug:
        print("===========================================================")
        print(f"[PROMPT]\n{inputs}\n\n[OUTPUT]\n{response}\n\n")
        pdb.set_trace()
    bar.next()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['medqa', 'medicaleval', 'ninth'], required=True)
    parser.add_argument("--input-file-name", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, required=True)
    
    parser.add_argument("--question-types", type=str, default=None)
    parser.add_argument("--cover", action="store_true", default=False)
    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    args.input_file_name = args.input_file_name.rsplit(".", 1)[0]
    args.output_file_name = args.output_file_name.rsplit(".", 1)[0]

    # mkdir
    # pdb.set_trace()
    mkdir(args.output_file_name)

    # data prepare process
    datas, _ = data_initialization(args)
    if datas == []:
        with open(f"{args.input_file_name}.json", "r", encoding="utf-8") as f:
            origin_questions = json.load(f)
            for q in origin_questions:
                q["patient_test"] = {str(i): {} for i in range(10)}
                datas.append(q)
                
    
    if args.debug:
        datas = datas[:10]
        args.workers = 1
        # datas[0]["history"] = []
        
    total_count = len(datas)

    if args.question_types is None:
        question_type_list = list(INSTRUCTIONS.keys())
    else:
        question_type_list = args.question_types.split(",")

    for hl in range(args.max_turn):
        for qt in question_type_list:
            ## Doctor genertaion
            bar = Bar(f'Question Type: {qt}, History Len: {hl}', max=total_count, suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
            pool = ThreadPool(processes=args.workers)
            pool.starmap(generate_qa, [[args, data, hl, qt, bar] for data in datas])
            pool.close()
            pool.join()
            bar.finish()

            if not args.debug:
                with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(datas, f, indent=4, ensure_ascii=False)


    if not args.debug:
        with open(f"{args.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)