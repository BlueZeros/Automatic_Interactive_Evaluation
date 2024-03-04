import os
import re
import pdb
import json
from openai import OpenAI
from utils.patient_conversation import get_patient_template, get_patient_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.general_utils import hisotry2str


class Agent:
    def __init__(self, args, model, conv):
        self.args = args
        self.model = model
        self.conv =conv
    
    def postprocess(self, outputs):
        outputs = outputs.split(self.conv.roles[0], 1)[0]
        outputs = outputs.split(self.conv.roles[1], 1)[0]
        
        parts = re.split(r'[A-Z]{3,}:[^:]', outputs)
        if len(parts) > 1:
            outputs = parts[0]

        return outputs.rstrip("\n " + self.conv.roles[0] + self.conv.roles[1])

    def generate(self):
        pass

    def log(self):
        self.model.log()


class Patient_Agent(Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
    
    def get_patient_info(self, whole_questions):
        if self.args.mode == "medqa":
            return whole_questions.rsplit(".", 1)[0]
        elif self.args.mode == "medicaleval":
            return split_chinese_medicalinfo_and_question(whole_questions)
        elif self.args.mode == "ninth":
            return whole_questions
    
    def generate(self, data, turn_id=0, bar=None):

        if len(data["history"]) <= self.args.max_turn:
            # pdb.set_trace()
            conv = self.conv.copy()
            conv.system_prompt_init(get_patient_prompt(self.args.patient_prompt_id, data["history"][-1]["state"]))
            
            conv.init_history(data["history"], first_key="doctor", second_key="patient")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt(patient_info=self.get_patient_info(data["raw_data"]["question"]))

            outputs = self.model.generate(prompt)
            outputs = self.postprocess(outputs, conv)
            data["history"][-1]["patient"] = outputs

            if self.args.debug:
                print(f"============== Patient Turn {turn_id}===================")
                print(f"Sample: {data['id']}")
                print(f"State: {data['history'][-1]['state']}")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}")
                pdb.set_trace()

        if bar is not None:
            bar.next()


class Doctor_Agent(Agent):
    def __init__(self, args, model, conv):
            super().__init__(args, model, conv)
    
    def postprocess(self, outputs):
        outputs = outputs.split(self.conv.roles[0], 1)[0]
        outputs = outputs.split(self.conv.roles[1], 1)[0]
        if "?" in outputs:
            outputs = outputs.split("?", 1)
            if len(outputs) > 1:
                outputs = outputs[0] + "?"
        elif "？" in outputs:
            outputs = outputs.split("？", 1)
            if len(outputs) > 1:
                outputs = outputs[0] + "？"
        
        # if self.args.mode == "medqa":
        parts = re.split(r'[A-Z]{4,}:[^:]', outputs)
        if len(parts) > 1:
            outputs = parts[0]

        return outputs.strip("\n :：" + self.conv.roles[0] + self.conv.roles[1])
    
    def generate(self, data, turn_id=0, bar=None):

        if len(data["history"]) < self.args.max_turn and (len(data["history"]) == 0 or "patient" in data["history"][-1].keys()):
            conv = self.conv.copy()
            conv.system_prompt_init(get_doctor_prompt(self.args.doctor_prompt_id))

            conv.init_history(data["history"], first_key="doctor", second_key="patient")
            conv.append_message(conv.roles[0], None)
            prompt = conv.get_prompt()

            outputs = self.model.generate(prompt)
            # if args.mode == "medqa":
            outputs = self.postprocess(outputs)
            data["history"].append({"doctor": outputs})
        
            if self.args.debug:
                print(f"============== Doctor Turn {turn_id}===================")
                print(f"Sample: {data['id']}")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}")
                pdb.set_trace()
        
        if bar is not None:
            bar.next()


class Doctor_Agent_V2(Doctor_Agent):
    def __init__(self, args, model, conv):
            super().__init__(args, model, conv)
    
    def generate(self, data, turn_id=0, bar=None):
        if len(data["history"]) < self.args.max_turn and (len(data["history"]) == 0 or data["history"][-1]["state"] != "D"):
            conv = self.conv.copy()
            conv.system_prompt_init(get_doctor_prompt(self.args.doctor_prompt_id))

            conv.init_history(data["history"], first_key="doctor", second_key="patient")
            conv.append_message(conv.roles[0], None)
            prompt = conv.get_prompt()

            outputs = ""
            while outputs == "":
                outputs = self.model.generate(prompt)
                if outputs == "":
                    print("Retrying...")
                    pdb.set_trace()
            # if args.mode == "medqa":
            outputs = self.postprocess(outputs)
            data["history"].append({"doctor": outputs})
        
            if self.args.debug:
                print(f"============== Doctor Turn {turn_id}===================")
                print(f"Sample: {data['id']}")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}")
                pdb.set_trace()

        if bar is not None:
            bar.next()

class Doctor_Agent_V3(Doctor_Agent):
    def __init__(self, args, model, conv):
            super().__init__(args, model, conv)
    
    def generate(self, data, turn_id=0, bar=None):
        if len(data["history"]) < self.args.max_turn and (len(data["history"]) == 0 or data["history"][-1]["state"] != "E") and (len(data["history"]) == 0 or "patient" in data["history"][-1]):
            conv = self.conv.copy()
            conv.system_prompt_init(get_doctor_prompt(self.args.doctor_prompt_id))

            conv.init_history(data["history"], first_key="doctor", second_key="patient")
            conv.append_message(conv.roles[0], None)
            prompt = conv.get_prompt()

            # outputs = ""
            # while outputs == "":
            #     outputs = self.model.generate(prompt)
            #     outputs = self.postprocess(outputs)

            #     if outputs == "":
            #         print("Retrying...")
            #         pdb.set_trace()
            outputs = self.model.generate(prompt)
            outputs = self.postprocess(outputs)
            data["history"].append({"doctor": outputs})
        
            if self.args.debug:
                print(f"============== Doctor Turn {turn_id}===================")
                print(f"Sample: {data['id']}")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}")
                pdb.set_trace()

        if bar is not None:
            bar.next()

class Patient_Agent_V2(Patient_Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
    
    def generate(self, data, turn_id=0, bar=None):
        # Four State Detection
        if len(data["history"]) <= self.args.max_turn and (data["history"][-1]["state"] != "D") and "patient" not in data["history"][-1].keys():
            state_prompt = get_patient_prompt(self.args.patient_prompt_id, data["history"][-1]["state"])
            if state_prompt: 
                conv = self.conv.copy()
                conv.system_prompt_init(state_prompt)
                
                conv.init_history(data["history"], turn=self.args.patient_history_len, first_key="doctor", second_key="patient")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt(patient_info=self.get_patient_info(data["raw_data"]["question"]))

                outputs = self.model.generate(prompt)
                outputs = self.postprocess(outputs)
                data["history"][-1]["patient"] = outputs

                if self.args.debug:
                    print(f"============== Patient Turn {turn_id}===================")
                    print(f"Sample: {data['id']}")
                    print(f"Prompt: {prompt}")
                    print(f"Output: {outputs}")
                    self.log()
                    pdb.set_trace()
        
        if bar is not None:
            bar.next()


class Patient_Agent_Test(Patient_Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
    
    def generate(self, data, question_type, history_len=0, bar=None):
        # pdb.set_trace()
        if "prediction" in data["patient_test"][str(history_len)][question_type].keys() and not self.args.cover:
            if bar is not None:
                bar.next()
            return 
        
        # pdb.set_trace()

        conv = self.conv.copy()
        conv.system_prompt_init(get_patient_prompt(self.args.patient_prompt_id))
        
        conv.init_history(data["history"], turn=history_len, latest=False, first_key="doctor", second_key="patient")
        conv.append_message(conv.roles[0], data["patient_test"][str(history_len)][question_type]["question"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(patient_info=self.get_patient_info(data["raw_data"]["question"]))

        outputs = self.model.generate(prompt)
        outputs = self.postprocess(outputs)
        data["patient_test"][str(history_len)][question_type]["prediction"] = outputs

        if self.args.debug:
            print(f"============== Patient Turn ===================")
            print(f"Sample: {data['id']}")
            print(f"Prompt: {prompt}")
            print(f"Output: {outputs}")
            pdb.set_trace()
        
        if bar is not None:
            bar.next()


class Patient_Agent_Test_W_State(Patient_Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
        # assert args.patient_model == "gpt4"
        self.state_agent = StateDetect_Agent_V3(args, model, conv, state_num=4)
    
    def memory_extraction(self, question, patient_info, state):

        if state not in ["A", "B"]:
            return state, ""

        if state[0] == "A":
            prompt = f"""
下面给出具体和宽泛的定义：
[具体]: 多为封闭式问题，带有一定的具体指向。询问症状时要提及具体的部位、症状、感受中的一个或多个。询问检查结果时，可以提及具体的部位、具体检查项目或者异常情况中的一个或多个。
[宽泛]: 多为开放式问题。指诸如询问类“有没有哪里不舒服？”，“哪里感觉很奇怪？”等完全没有具体条件的情况均属于此类宽泛。

<问题>：{question}
请判断在<问题>中医生是否有询问患者[具体]的医疗信息或者给出[具体]的建议。如果有，直接输出[具体]。如果没有，则直接输出[宽泛]。
"""     
        else:
            prompt = f"""
下面给出具体和宽泛的定义：
[具体]: 需要有具体的内容。比如检查要给出某种特定的检查项目，治疗要给出具体方案，药物要给出具体名称等。
[宽泛]: 只建议做检查治疗，但没有具体的检查项目和治疗方案时，则为宽泛。

<问题>：{question}
请判断在<问题>中医生是否有询问患者[具体]的医疗信息或者给出[具体]的建议。如果有，直接输出[具体]。如果没有，则直接输出[宽泛]。
"""
        question_extraction = self.model.generate(prompt)
        if self.args.debug:
            print(f"============== Patient question extraction ===================")
            print(f"Prompt: {prompt}")
            print(f"Output: {question_extraction}")
            pdb.set_trace()
        if question_extraction == "[宽泛]" or question_extraction == "宽泛":
            state += "-B"
            return state, ""
        else:
            state += "-A"
        
        if state[0] == "A":
            prompt = f"""
下面给出有无相关信息的定义：
[有相关信息]: <患者信息>中有<问题>所询问的信息，包括描述患者有该症状或者没有该症状只要有相关内容都算在此类。
[无相关信息]: <患者信息>中没有<问题>所询问的信息，<患者信息>没有相关信息的都算在此类。

<患者信息>：{patient_info}

<问题>: {question}
请判断在<患者信息>中是否有<问题>中所询问的相关信息，如果[有相关信息]，则直接输出相关的文本语句，注意不要输出不相关的内容。如果[无相关信息]，则直接输出[无相关信息]。
"""
        else:
            prompt = f"""
下面给出有无相关信息的定义：
[有相关信息]: <患者信息>中有<问题>所建议的结果信息，包括任何检查项目与治疗方案有相关的结果的都算在此类。
[无相关信息]: <患者信息>中没有<问题>所建议的结果信息，包括完全没有提到相关检查项目和治疗方案或者没有对应的相关结果的都算在此类。

<患者信息>：{patient_info}

<问题>: {question}
请判断在<患者信息>中是否有<问题>中建议措施的相关信息，如果[有相关信息]，则直接输出相关的文本语句，注意不要输出不相关的内容。如果[无相关信息]，则直接输出[无相关信息]。
"""
        memory_extraction = self.model.generate(prompt)
        if self.args.debug:
            print(f"============== Patient memory extraction ===================")
            print(f"Prompt: {prompt}")
            print(f"Output: {memory_extraction}")
            pdb.set_trace()
        
        if memory_extraction == "[无相关信息]" or memory_extraction == "无相关信息":
            state += "-B"
            return state, ""
        else:
            state += "-A"
            return state, memory_extraction
    
    def generate(self, data, question_type, history_len=0, bar=None):
        if "prediction" in data["patient_test"][str(history_len)][question_type].keys() and not self.args.cover and not self.args.debug:
            if bar is not None:
                bar.next()
            return 

        conv = self.conv.copy()

        if self.args.golden_state:
            state = question_type[0]
        else:
            state = self.state_agent.generate(question=data["patient_test"][str(history_len)][question_type]["question"])

        state, memory = self.memory_extraction(data["patient_test"][str(history_len)][question_type]["question"], self.get_patient_info(data["raw_data"]["question"]), state)
        data["patient_test"][str(history_len)][question_type]["state_prediction"] = state
        data["patient_test"][str(history_len)][question_type]["memory"] = memory

        conv.system_prompt_init(get_patient_prompt(self.args.patient_prompt_id, state))
        
        conv.init_history(data["history"], turn=history_len, latest=False, first_key="doctor", second_key="patient")
        conv.append_message(conv.roles[0], data["patient_test"][str(history_len)][question_type]["question"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(patient_info=memory)

        outputs = self.model.generate(prompt)
        outputs = self.postprocess(outputs)
        data["patient_test"][str(history_len)][question_type]["prediction"] = outputs

        if self.args.debug:
            print(f"============== Patient Turn ===================")
            print(f"Sample: {data['id']}")
            print(f"State: {state}")
            print(f"Prompt: {prompt}")
            print(f"Output: {outputs}")
            pdb.set_trace()
        
        if bar is not None:
            bar.next()

class Patient_Agent_Test_W_State_V2(Patient_Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
        # assert args.patient_model == "gpt4"
        self.state_agent = StateDetect_Agent_V4_Test(args, model, conv, state_num=4)
    
    def generate(self, data, question_type, history_len=0, bar=None):
        if "prediction" in data["patient_test"][str(history_len)][question_type].keys() and not self.args.cover and not self.args.debug:
            if bar is not None:
                bar.next()
            return 

        conv = self.conv.copy()

        state = data["patient_test"][str(history_len)][question_type]["state_prediction"]
        memory = data["patient_test"][str(history_len)][question_type]["memory"]

        conv.system_prompt_init(get_patient_prompt(self.args.patient_prompt_id, state))
        
        conv.init_history(data["history"], turn=history_len, latest=False, first_key="doctor", second_key="patient")
        conv.append_message(conv.roles[0], data["patient_test"][str(history_len)][question_type]["question"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(patient_info=memory)

        outputs = self.model.generate(prompt)
        outputs = self.postprocess(outputs)
        data["patient_test"][str(history_len)][question_type]["prediction"] = outputs

        if self.args.debug:
            print(f"============== Patient Turn ===================")
            print(f"Sample: {data['id']}")
            print(f"State: {state}")
            print(f"Memory: {memory}")
            print(f"Prompt: {prompt}")
            print(f"Output: {outputs}")
            pdb.set_trace()
        
        if bar is not None:
            bar.next()


class Patient_Agent_V3(Patient_Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
    
    def generate(self, data, turn_id=0, bar=None):
        # Four State Detection
        if len(data["history"]) <= self.args.max_turn and (data["history"][-1]["state"] != "E") and "patient" not in data["history"][-1].keys():
            state_prompt = get_patient_prompt(self.args.patient_prompt_id, data["history"][-1]["state"])
            if state_prompt: 
                conv = self.conv.copy()
                conv.system_prompt_init(state_prompt)
                
                conv.init_history(data["history"], turn=self.args.patient_history_len, first_key="doctor", second_key="patient")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt(patient_info=self.get_patient_info(data["history"][-1]["memory"]))

                outputs = self.model.generate(prompt)

                outputs = self.postprocess(outputs)
                data["history"][-1]["patient"] = outputs

                if self.args.debug:
                    print(f"============== Patient Turn {turn_id}===================")
                    print(f"Sample: {data['id']}")
                    print(f"State: {data['history'][-1]['state']}")
                    print(f"Memory: {data['history'][-1]['memory']}")
                    print(f"Prompt: {prompt}")
                    print(f"Output: {outputs}")
                    self.log()
                    pdb.set_trace()
        
        if bar is not None:
            bar.next()


class StateDetect_Agent(Agent):
    def __init__(self, args, model, conv=None, state_num=4):
        super().__init__(args, model, conv)

        self.state_num = state_num
        self.state_descripts = {"A": "口头信息询问", "B": "医疗检查询问", "C": "外科检查询问", "D": "无询问"}
        self.logit_bias = {(32+i):100 for i in range(state_num)}

        self.prompt = "在问诊过程中，医生的问题可以被分为四个类型:\n\
    (A) 基础信息询问：对于这类问题病人可以根据自身的情况直接回答，询问的信息包括但不限于个人信息，医疗历史，主观感受，以及简单的外伤描述等。\n\
    (B) 检查询信息问：这类问题一般涉及到医疗检查和化验结果，询问的信息包括但不限于MRI，CT，细胞活检等。建议去医院做检查也属于此类\n\
    (C) 物理动作询问：病人需要作出某些物理实体动作才能完成医生的要求，如张嘴，侧躺，站立等。\n\
    (D) 无询问：医生已经完成问诊，不需要病人进行回复。\n\n\
请根据以上各问题类型的描述从所给类型中给出下列医生问题的类型:\n\n\
医生问题: {question}\n\
问题类型: ("
        
    def generate(self, data, turn_id, bar=None):
        if "state" not in data["history"][-1].keys():
            assert "patient" not in data["history"][-1].keys()

            if turn_id == -1:
                data["history"][-1]["state"] = "A"
            else:
                question = data["history"][-1]["doctor"]
                prompt = self.prompt.format(question=question)
                outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)

                data["history"][-1]["state"] = outputs

                if self.args.debug:
                        print(f"============== State Turn {turn_id}===================")
                        print(f"Sample: {data['id']}")
                        print(f"Prompt: {prompt}")
                        print(f"Output: {outputs}: {self.state_descripts[outputs]}")
                        pdb.set_trace()
        
        if bar is not None:
            bar.next()

class StateDetect_Agent_V2(Agent):
    def __init__(self, args, model, conv=None, state_num=4):
        super().__init__(args, model, conv)

        self.state_num = state_num
        self.state_descripts = {"A": "口头信息询问", "B": "医疗检查询问", "C": "外科检查询问", "D": "无询问"}
        self.logit_bias = {(32+i):100 for i in range(state_num)}
        
        self.prompt = "在问诊过程中，医生的问题可以被分为四个类型:\n\
    (A) 基础信息询问：对于这类问题病人可以根据自身的情况直接回答，询问的信息包括但不限于个人信息，医疗历史，主观感受，以及简单的外伤描述等。\n\
    (B) 检查询信息问：这类问题一般涉及到医疗检查和化验结果，询问的信息包括但不限于MRI，CT，细胞活检等。建议去医院做检查也属于此类\n\
    (C) 物理动作询问：病人需要作出某些物理实体动作才能完成医生的要求，如张嘴，侧躺，站立等。\n\
    (D) 无询问：医生已经完成问诊，不需要病人进行回复。\n\n\
请根据以上各问题类型的描述从所给类型中给出下列医生问题的类型:\n\n\
医生问题: {question}\n\
问题类型: ("
        
    def generate(self, data, turn_id, bar=None):
        if "state" not in data["history"][-1].keys():
            assert "patient" not in data["history"][-1].keys()

            if turn_id == -1:
                data["history"][-1]["state"] = "A"
            else:
                question = data["history"][-1]["doctor"]
                prompt = self.prompt.format(question=question)
                outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)

                data["history"][-1]["state"] = outputs

                if self.args.debug:
                        print(f"============== State Turn {turn_id}===================")
                        print(f"Sample: {data['id']}")
                        print(f"Prompt: {prompt}")
                        print(f"Output: {outputs}: {self.state_descripts[outputs]}")
                        pdb.set_trace()
        
        if bar is not None:
            bar.next()

class StateDetect_Agent_V3(Agent):
    def __init__(self, args, model, conv=None, state_num=5):
        super().__init__(args, model, conv)

        self.state_num = state_num
        self.state_descripts = {"A": "信息询问类", "B": "医疗建议类", "C": "物理动作类", "D": "其他话题类", "E":"对话结束类"}
        self.logit_bias = model.get_logit_bias(state_num)
        
        self.prompt = "在问诊过程中，医生的问题可以被分为五个类型:\n\
    (A) 信息询问类：医生询问病人医疗疾病相关症状信息，一般问题中带有'?'或'？'且不属于(C)(D)类的都属于此类。\n\
    (B) 医疗建议类：医生建议病人去医院就诊，做检查，或者给出某些治疗方案等等。问题中有关键词'建议'的都属于此类。\n\
    (C) 物理动作类：医生要求患者做出某些动作以观察，配合，感受等，动作包括但不限于张嘴，侧躺，站立，按压等都属于此类。医生要求上传图片等网络操作等也属于此类\n\
    (D) 其他话题类：医生的问题不属于医疗问诊的情景下，询问的主题与医疗疾病无关的的都属于此类，包括但不限于：爱好，电影，美食等等。\n\
    (E) 无询问：医生已经完成问诊，不需要病人进行回复。\n\n\
请根据以上各问题类型的描述从所给类型中选出下列医生问题最属于的类型:\n\n\
医生问题: {question}\n\
问题类型: ("
        
    def generate(self, question):
        prompt = self.prompt.format(question=question)
        outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)

        if self.args.debug:
                print(f"============== State Detection ===================")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}: {self.state_descripts[outputs]}")
                pdb.set_trace()
        
        return outputs

class StateDetect_Agent_V4(Agent):
    def __init__(self, args, model, conv=None, state_num=5):
        super().__init__(args, model, conv)

        self.state_num = state_num
        self.state_descripts = {"A": "信息询问类", "B": "医疗建议类", "C": "物理动作类", "D": "其他话题类", "E":"对话结束类"}
        self.logit_bias = model.get_logit_bias(state_num)
        
        if args.mode == "ninth":
            self.stageI_prompt = """
在问诊过程中，医生的问题可以被分为五个类型:\n\
    (A) 信息询问类：医生询问病人医疗疾病相关症状信息，一般问题中带有'?'或'？'且不属于(C)(D)类的都属于此类。\n\
    (B) 医疗建议类：医生建议病人去医院就诊，做检查，或者给出某些治疗方案等等。问题中有关键词'建议'的都属于此类。\n\
    (C) 物理动作类：医生要求患者做出某些动作以观察，配合，感受等，动作包括但不限于张嘴，侧躺，站立，按压等都属于此类。\n\
    (D) 其他话题类：医生的问题不属于医疗问诊的情景下，询问的主题与医疗疾病无关的的都属于此类，包括但不限于：爱好，电影，美食等等。\n\
    (E) 无询问：医生已经完成问诊，不需要病人进行回复。\n\n\
请根据以上各问题类型的描述从所给类型中选出下列医生问题最属于的类型:\n\n\
医生问题: {question}\n\
问题类型: ("""

            self.stageII_prompt = {
            "A": """
<定义>：
    [具体]: <问题>带有一定的具体指向。询问症状时至少要询问具体的的部位、症状、感受、情况中的一个。询问检查结果时至少提及具体的部位、具体检查项目或者异常情况中的一个。注意，如果是询问特定的医疗情况，如就医史，家族史，慢性病史，手术史等则无论如何都属于[具体]。此外，如果问题中出现指示代词，这说明该问题与对话历史相承接，也应该属于具体。
    [宽泛]: <问题>诸如“哪里不舒服？”，“哪里感觉很奇怪？”， “还有什么不舒服么？”等完全没有具体信息指向的询问均属于[宽泛]。

<问题>：{question}

请根据<定义>判断在<问题>中医生是否有询问患者[具体]的医疗信息或者给出[具体]的建议。如果有，直接输出[具体]。如果没有，则直接输出[宽泛]。
""",
            "B": """
<定义>：
    [具体]: <建议>中含有具体的医疗名词或者具体的建议措施。如具体的检查检查类型（包括但不限于X光，MRI，细胞活检等），具体治疗方案（包括但不限于具体的手术治疗，运动，饮食等），具体药物类型等，以及生活方面的具体建议等。
    [宽泛]: <建议>中宽泛的给出的建议，但是没有具体的检查项目和治疗方案药物类型生活建议时属于[宽泛]。

<建议>：{question}

请根据<定义>判断在<建议>中医生是否有询问患者[具体]的医疗信息或者给出[具体]的建议。如果有，直接输出[具体]。如果没有，则直接输出[宽泛]。
"""}

            self.stageIII_prompt = {
            "A":"""
<定义>：
    [有相关信息]: <患者信息>中有<问题>所询问的信息，包括描述患者有该症状或者没有该症状只要有相关内容都算在此类。
    [无相关信息]: <患者信息>中没有<问题>所询问的信息，<患者信息>没有相关信息的都算在此类。

<患者信息>：{patient_info}

<问题>: {question}

请根据<定义>判断在<患者信息>中是否有<问题>中所询问的相关信息，如果[有相关信息]，则直接输出相关的文本语句，注意不要输出不相关的内容。如果[无相关信息]，则直接输出[无相关信息]。
""",
            "B":"""
<定义>：
    [有相关信息]: <患者信息>中有<建议>中所建议检查或治疗方案的的结果信息，包括任何检查项目与治疗方案有相关的结果的都算在此类。
    [无相关信息]: <患者信息>中没有<建议>中所建议检查或治疗方案的的结果信息，包括完全没有提到相关检查项目和治疗方案或者没有对应的相关结果的都算在此类。

<患者信息>：{patient_info}

<建议>: {question}

请根据<定义>判断在<患者信息>中是否有<建议>中建议措施的相关信息，如果[有相关信息]，则直接输出相关的文本语句，注意不要输出不相关的内容。如果[无相关信息]，则直接输出[无相关信息]。
"""}
            
        elif args.mode == "medqa":
            self.stageI_prompt =  """
During the consultation process, a doctor's questions can be categorized into five types:\n\
    (A) Inquiry: The doctor asks the patient for medical and disease-related symptom information. Generally, questions containing '?' or '？' that do not belong to categories (C) or (D) are included here.\n\
    (B) Advice: The doctor advises the patient to visit a hospital for consultation, undergo examinations, or suggests certain treatment plans, etc. Questions containing the keyword 'advice' fall into this category.\n\
    (C) Demand: The doctor requests the patient to perform certain actions for observation, cooperation, sensation, etc. Actions include, but are not limited to, opening the mouth, lying on the side, standing, pressing, etc.\n\
    (D) Other Topics: Questions from the doctor that do not pertain to the medical consultation scenario, and are unrelated to medical diseases, such as hobbies, movies, cuisine, etc.\n\
    (E) End: The doctor has completed the consultation and does not require a response from the patient.\n\n\
Based on the descriptions of each question type above, identify the most appropriate category for the following doctor's question:\n\n\
Doctor's Question: {question}\n\
Question Type: ("""

            self.stageII_prompt = {
            "A": """
<Definition>:
    [Specific]: <Question> has a certain specific direction. When asking about symptoms, it should at least inquire about specific body parts, symptoms, sensations, or situations. When asking about examination results, it should mention specific body parts, specific examination items, or abnormal situations. Note that if it's about specific medical conditions, like medical history, family history, chronic illnesses, surgical history, etc., they are always considered [Specific]. Specifically, if the <Question> contain about demonstrative like "these" or "this", then it is related to the above and should belongs to the [Specific]
    [Broad]: <Question> such as "Where do you feel uncomfortable?" or "Where does it feel strange?" without any specific information direction are considered [Broad].

<Question>: {question}

Based on the <Definition>, determine whether the doctor's <Question> asks for [Specific] medical information from the patient or gives [Specific] advice. If so, directly output [Specific]. If not, output [Broad].
""",
            "B": """
<Definition>:
    [Specific]: <Advice> contains specific types of examinations or test (including but not limited to X-rays, MRI, biopsy, etc.), specific treatment plans (including but not limited to specific surgical treatments, exercises, diets, etc.), specific types of medication, etc.
    [Broad]: <Advice> broadly given without any specific examination/test, treatment plans, doctor's orders, exercises, diets and medication types is considered [Broad]. As long as any of the above information appears, <Advice> does not fall into this category.

<Advice>: {question}

Based on the <Definition>, determine whether the doctor's <Advice> asks for [Specific] medical information from the patient or gives [Specific] advice. If so, directly output [Specific]. If not, output [Broad].
"""}

            self.stageIII_prompt = {
            "A":"""
<Definition>:
    [Relevant Information]: <Patient Information> contains information asked in <Question>, including descriptions of having or not having the symptom, as long as there's relevant content.
    [No Relevant Information]: <Patient Information> does not contain information asked in <Question>, and there's no relevant content in the information.

<Patient Information>: {patient_info}

<Question>: {question}

Based on the <Definition>, determine whether <Patient Information> contains relevant information asked in <Question>. If [Relevant Information] is present, directly output the relevant text statement, ensuring not to include irrelevant content. If [No Relevant Information], then directly output [No Relevant Information].
""",
            "B":"""
<Definition>:
    [Relevant Information]: <Patient Information> contains results of the examinations or treatment plans suggested in <Advice>, including any results related to the suggested examination items and treatment plans.
    [No Relevant Information]: <Patient Information> does not contain results of the examinations or treatment plans suggested in <Advice>, including no mention of relevant examination items and treatment plans or no corresponding results.

<Patient Information>: {patient_info}

<Advice>: {question}

Based on the <Definition>, determine whether <Patient Information> contains relevant information about the measures suggested in <Advice>. If [Relevant Information] is present, directly output the relevant text statement, ensuring not to include irrelevant content. If [No Relevant Information], then directly output [No Relevant Information].
"""}

        else:
            raise NotImplementedError

    def generate_stageI(self, data, turn_id, bar=None):
        if "state" not in data["history"][-1].keys():
            question = data["history"][-1]["doctor"]
            prompt = self.stageI_prompt.format(question=question)
            outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)

            data["history"][-1]["state"] = outputs

            # if self.args.debug:
            #         print(f"============== Detect StageI Turn {turn_id}===================")
            #         print(f"Sample: {data['id']}")
            #         print(f"Prompt: {prompt}")
            #         print(f"Output: {outputs}: {self.state_descripts[outputs]}")
            #         pdb.set_trace()
        
        if bar is not None:
            bar.next()
    
    def generate_stageII(self, data, turn_id, bar=None):
        if data["history"][-1]["state"] not in ["A", "B"]:
            return
        
        question = data["history"][-1]["doctor"]

        prompt = self.stageII_prompt[data["history"][-1]["state"]].format(question=question)
        outputs = self.model.generate(prompt)

        if self.args.mode == "ninth":
            if outputs == "[宽泛]" or outputs == "宽泛":
                data["history"][-1]["state"] += "-B"
            elif outputs == "[具体]" or outputs == "具体":
                data["history"][-1]["state"] += "-A"
            else:
                data["history"][-1]["state"] += "-B"

        elif self.args.mode == "medqa":
            if outputs == "[Broad]" or outputs == "Broad":
                data["history"][-1]["state"] += "-B"
            elif outputs == "[Specific]" or outputs == "Specific":
                data["history"][-1]["state"] += "-A"
            else:
                # print("stageII output error!")
                # pdb.set_trace()
                data["history"][-1]["state"] += "-B"
        else:
            raise NotImplementedError
        
        # if self.args.debug:
        #         print(f"============== Detect StageII Turn {turn_id}===================")
        #         print(f"Sample: {data['id']}")
        #         print(f"Prompt: {prompt}")
        #         print(f"Output: {outputs}")
        #         pdb.set_trace()
        
        if bar is not None:
            bar.next()
    
    def generate_stageIII(self, data, turn_id, bar=None):
        if data["history"][-1]["state"] not in ["A-A", "B-A"]:
            if "memory" not in data["history"][-1].keys():
                data["history"][-1]["memory"] = ""
            return
        
        question = data["history"][-1]["doctor"]
        patient_info = data["raw_data"]["question"]

        prompt = self.stageIII_prompt[data["history"][-1]["state"][0]].format(patient_info=patient_info, question=question)
        outputs = self.model.generate(prompt)

        if self.args.mode == "ninth":
            if outputs == "[无相关信息]" or outputs == "无相关信息":
                data["history"][-1]["state"] += "-B"
                data["history"][-1]["memory"] = ""
            else:
                data["history"][-1]["state"] += "-A"
                data["history"][-1]["memory"] = outputs
        elif self.args.mode == "medqa":
            if outputs == "[No Relevant Information]" or outputs == "No Relevant Information":
                data["history"][-1]["state"] += "-B"
                data["history"][-1]["memory"] = ""
            else:
                data["history"][-1]["state"] += "-A"
                data["history"][-1]["memory"] = outputs
        else:
            raise NotImplementedError

        # if self.args.debug:
        #         print(f"============== Detect StageIII Turn {turn_id}===================")
        #         print(f"Sample: {data['id']}")
        #         print(f"Prompt: {prompt}")
        #         print(f"Output: {outputs}")
        #         pdb.set_trace()
        
        if bar is not None:
            bar.next()
        
    def generate(self, data, turn_id, detect_type="stageI", bar=None):
        assert detect_type in ["stageI", "stageII", "stageIII"], f"detect_type: {detect_type} is not defined!"

        if detect_type == "stageI":
            self.generate_stageI(data, turn_id, bar)
        elif detect_type == "stageII":
            self.generate_stageII(data, turn_id, bar)
        elif detect_type == "stageIII":
            self.generate_stageIII(data, turn_id, bar)

        if self.args.debug:
            self.log()

class StateDetect_Agent_V4_Test(StateDetect_Agent_V4):
    def __init__(self, args, model, conv=None, state_num=4):
        super().__init__(args, model, conv, state_num)
        # pdb.set_trace()
        self.stageII_prompt = {
            "A": """
<定义>：
    (A)[具体]: <问题>带有一定的具体指向。询问症状时至少要询问具体的的部位、症状、感受、情况中的一个。询问检查结果时至少提及具体的部位、具体检查项目或者异常情况中的一个。注意，如果是询问特定的医疗情况，如就医史，家族史，慢性病史，手术史等则无论如何都属于[具体]。
    (B)[宽泛]: <问题>诸如“哪里不舒服？”，“哪里感觉很奇怪？”等完全没有具体信息指向的询问均属于[宽泛]。

<问题>：{question}

根据<定义>中的描述，<问题>属于(""",
            "B": """
<定义>：
    (A)[具体]: <建议>中含有具体的检查类型（包括但不限于X光，MRI，细胞活检等），具体治疗方案（包括但不限于具体的手术治疗，运动，饮食等），具体药物类型等。
    (B)[宽泛]: <建议>中宽泛的给出的建议，没有具体的检查项目和治疗方案药物类型时属于[宽泛]。

<建议>：{question}

根据<定义>中的描述，<建议>属于("""}
    
    def generate_stageI(self, data, question_type, history_len, bar=None):
        if "state_prediction" in data["patient_test"][str(history_len)][question_type] and not self.args.cover:
            return
        prompt = self.stageI_prompt.format(question=data["patient_test"][str(history_len)][question_type]["question"])
        outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)
        data["patient_test"][str(history_len)][question_type]["state_prediction"] = outputs

        # if self.args.debug:
        #     print(f"============== Detect StageI ===================")
        #     print(f"Sample: {data['id']}")
        #     print(f"Prompt: {prompt}")
        #     print(f"Output: {outputs}: {self.state_descripts[outputs]}")
        #     pdb.set_trace()
        
        if bar is not None:
            bar.next()
    
    def generate_stageII(self, data, question_type, history_len, bar=None):
        if data["patient_test"][str(history_len)][question_type]["state_prediction"] not in ["A", "B"]:
            return

        # pdb.set_trace()
        prompt = self.stageII_prompt[data["patient_test"][str(history_len)][question_type]["state_prediction"]].format(question=data["patient_test"][str(history_len)][question_type]["question"])
        outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.model.get_logit_bias(2))

        data["patient_test"][str(history_len)][question_type]["state_prediction"] += f"-{outputs}"

        if self.args.debug:
            print(f"============== Detect StageII ===================")
            print(f"Sample: {data['id']}")
            print(f"Prompt: {prompt}")
            print(f"Output: {outputs}")
            pdb.set_trace()
        
        if bar is not None:
            bar.next()
    
    def generate_stageIII(self, data, question_type, history_len, bar=None):
        if data["patient_test"][str(history_len)][question_type]["state_prediction"] not in ["A-A", "B-A"]:
            if "memory" not in data["patient_test"][str(history_len)][question_type].keys():
                data["patient_test"][str(history_len)][question_type]["memory"] = ""
            return
        
        question = data["patient_test"][str(history_len)][question_type]["question"]
        patient_info = data["raw_data"]["question"]

        prompt = self.stageIII_prompt[data["patient_test"][str(history_len)][question_type]["state_prediction"][0]].format(patient_info=patient_info, question=question)
        outputs = self.model.generate(prompt)

        if outputs == "[无相关信息]" or outputs == "无相关信息":
            data["patient_test"][str(history_len)][question_type]["state_prediction"] += "-B"
            data["patient_test"][str(history_len)][question_type]["memory"] = ""
        else:
            data["patient_test"][str(history_len)][question_type]["state_prediction"] += "-A"
            data["patient_test"][str(history_len)][question_type]["memory"] = outputs
        
        # if self.args.debug:
        #     print(f"============== Detect StageI ===================")
        #     print(f"Sample: {data['id']}")
        #     print(f"Prompt: {prompt}")
        #     print(f"Output: {outputs}")
        #     pdb.set_trace()

        if bar is not None:
            bar.next()

    def generate(self, data, question_type, history_len=0, detect_type="stageI", bar=None):
        assert detect_type in ["stageI", "stageII", "stageIII"], f"detect_type: {detect_type} is not defined!"

        if detect_type == "stageI":
            self.generate_stageI(data, question_type, history_len, bar)
        elif detect_type == "stageII":
            self.generate_stageII(data, question_type, history_len, bar)
        elif detect_type == "stageIII":
            self.generate_stageIII(data, question_type, history_len, bar)

        if self.args.debug:
            self.log()

class Dignosis_Agent(Agent):
    def __init__(self, args, model, conv=None, candidates_num=5):
        super().__init__(args, model, conv)
        self.state_num = candidates_num
        self.logit_bias = {(32+i):100 for i in range(candidates_num)}
    
        self.prompt = "Please answer the following question accoding to the consultation conversation between the doctor and the patient.\n\
**Conversation:**\n{conversations}\n\
**Question:** {question}\n{options}\n\
**Answer:**("

    def generate(self, data, bar=None):

        question = data["question"]
        options = data["raw_data"]["options"]
        conversations = hisotry2str(data["history"])
        prompt = self.prompt.format(conversations=conversations, question=question, options=options)
        outputs = self.model.multiple_choice_selection(prompt, logit_bias=self.logit_bias)

        data["diagnosis_self"] = outputs

        if self.args.debug:
                    print(f"============== Diagnosis===================")
                    print(f"Sample: {data['id']}")
                    print(f"Prompt: {prompt}")
                    print(f"Output: {outputs}")
                    pdb.set_trace()

        if bar is not None:
            bar.next()
    

class EvalAgent(Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
        if args.eval_type == "patient":
            self.prompt = "<model1>\n{model1_dialogue}\n<\model1>\n\n<model2>\n{model2_dialogue}\n<\model2>\n\
以上是model1和model2两个医生模型与同一患者的对话，请从患者角度根据以下指标选择你认为更好的医生模型。对每个维度如果model1更好直接输出<model1>，model2更好则输出<model2>，一样好或者一样坏则输出<tie>。注意，请直接输出结果，不要输出解释。\n\
1. Effectiveness: Which model (1 or 2) provided more beneficial advice or diagnosis in general terms?\n\
2. Clarity: Which model (1 or 2) communicated more clearly and was easier to understand, particularly in explaining any medical or technical terms?\n\
3. Understanding: Which model (1 or 2) showed greater consideration of the patient's preferences and engagement with the patient’s ideas or concerns? \n\
4. Empathy: Which model (1 or 2) demonstrated more empathy and a better response to the patient’s emotional state and thoughts?\n\
5. Conclusion: Which model (1 or 2) appeared more credible, reliable, and professional overall?"

        elif args.eval_type == "doctor":
            self.prompt = "<model1>\n{model1_dialogue}\n<\model1>\n\n<model2>\n{model2_dialogue}\n<\model2>\n\
以上是model1和model2两个医生模型与同一患者的对话，请从医生角度根据以下指标选择你认为更好的医生模型。对每个维度如果model1更好直接输出<model1>，model2更好则输出<model2>，一样好或者一样坏则输出<tie>。注意，请直接输出结果，不要输出解释。\n\
1. Information Gathering: Which model (1 or 2) more effectively collects key patient information, including chief complaints and illness history?\n\
2. Inquiry Logic: Assess which model (1 or 2) has a more logical and non-repetitive questioning approach.\n\
3. Diagnosis and Recommendations: Determine which model (1 or 2) makes accurate diagnoses with adequate information and provides appropriate advice when information is scarce.\n\
4. Humanistic Care: Evaluate which model (1 or 2) better demonstrates empathy, respect, and support for the patient's emotional and psychological needs.\n\
5. Conclusion: Decide which model (1 or 2) excels overall, considering their information gathering, inquiry logic, diagnostic accuracy, and humanistic care."

        else:
            raise NotImplementedError

    def generate(self, data, bar=None):
        if "result" in data:
            if bar is not None:
                bar.next()
            return
        
        conv = self.conv.copy()
        conv.init_history(data["model1"]["results"]["history"], first_key="doctor", second_key="patient")
        model1_dialogue = conv.get_prompt()

        conv = self.conv.copy()
        conv.init_history(data["model2"]["results"]["history"], first_key="doctor", second_key="patient")
        model2_dialogue = conv.get_prompt()

        inputs = self.prompt.format(model1_dialogue=model1_dialogue, model2_dialogue=model2_dialogue)
        outputs = self.model.generate(inputs)

        data["result"] = outputs

        if self.args.debug:
            print(f"============== Evaluation ===================")
            print(f"Sample: {data['id']}")
            print(f"Prompt: {inputs}")
            print(f"Output: {outputs}")
            pdb.set_trace()

        if bar is not None:
            bar.next()