import dataclasses
import torch
from enum import auto, Enum
from utils.general_utils import get_value
from typing import List, Tuple, Any
import pdb


class SeparatorStyle(Enum):
    """Different separator style."""
    VICUNA = auto()
    LLAMA2 = auto()
    FALCON = auto()
    CHATINTERN = auto()
    CHATGLM3 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    stop_ids: List[torch.tensor]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.VICUNA
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None
    
    def system_prompt_init(self, prompt):
        self.system = self.system.format(prompt=prompt)

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.VICUNA:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.FALCON:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "

            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += role + ": " + message + " "
                    else:
                        ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ": " + message + seps[i % 2] + "\n"
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def init_history(self, history_list, turn=-1, latest=True, first_key="question", second_key="answer"):
        if turn < 0 or turn > len(history_list):
            pass
        elif turn == 0:
            history_list = []
        else:
            if latest:
                history_list = history_list[-turn:]
            else:
                history_list = history_list[:turn]

        for history in history_list:
            if first_key in history.keys():
                self.messages.append([self.roles[0], history[first_key]])
                if second_key in history.keys():
                    self.messages.append([self.roles[1], history[second_key]])
    
    def pop_message(self):
        self.messages = self.messages[:-1]
    
    def clean_message(self):
        self.messages = []

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            stop_ids=self.stop_ids,
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

conv_vicuna_v1_1 = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([3970, 1783, 1955]), torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 29925, 1299, 29902, 3919]), torch.tensor([349, 1299, 29902, 3919]), torch.tensor([13, 13, 29925, 1299, 29902, 3919]),]
)

conv_chatgpt = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["DOCTOR", "PATIENT"]
)

conv_chatgpt_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["[医生]", "[患者]"]
)

conv_bloomz_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[torch.tensor([62, 25967, 64]), torch.tensor([62, 26122, 64])]
)

conv_gpt4 = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=["DOCTOR", "PATIENT"]
)

conv_gpt4_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[]
)

conv_falcon = Conversation(
        system="System: {prompt}",
        roles=("DOCTOR", "PATIENT"),
        messages=[],
        sep_style=SeparatorStyle.FALCON,
        offset=0,
        sep="\n",
        sep2="<|endoftext|>",
        stop_ids=[torch.tensor([4310, 4274, 1951]), torch.tensor([8769, 4274, 1951]), torch.tensor([40392]), torch.tensor([19363]), torch.tensor([38293]), torch.tensor([12775])]
    )

conv_llama2 = Conversation(
    system="[INST] <<SYS>>\n{prompt}<</SYS>>\n\n",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    sep_style=SeparatorStyle.LLAMA2,
    offset=0,
    sep=" ",
    sep2=" </s><s>",
    stop_ids=[torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 3970, 1783, 1955]), torch.tensor([29871, 11662, 1783, 1955]), torch.tensor([13,  3970,  1783,  1955, 29901]), torch.tensor([ 13, 29925, 1299, 29902, 3919]), torch.tensor([1299, 29902, 3919])]
)

conv_llama2_zh = Conversation(
    system="[INST] <<SYS>>\n{prompt}<</SYS>>\n\n",
    roles=("[医生]", "[患者]"),
    messages=[],
    sep_style=SeparatorStyle.LLAMA2,
    offset=0,
    sep=" ",
    sep2=" </s><s>",
    stop_ids=[torch.tensor([232, 143, 190, 30486])]
)

conv_baichuan = Conversation(
    system="{prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([5946])]
)

conv_baichuan_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep=" ",
    sep2="</s>",
    stop_ids=[torch.tensor([1633, 5946, 31295]), torch.tensor([1633, 4304, 31295])]
)

conv_internlm = Conversation(
    system="{prompt}",
    roles=("[DOCTOR]", "[PATIENT]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([68305])]
)

conv_internlm_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([336, 68305, 332]), torch.tensor([336, 68049, 332])]
)

conv_chatglm3 = Conversation(
    system="<|system|>\n {prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATGLM3,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([[790, 32718, 30996]]), torch.tensor([[790, 32016, 30996]])]
)
conv_chatglm3_zh = Conversation(
    system="<|system|>\n {prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATGLM3,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([[790, 32718, 30996]]), torch.tensor([[790, 32016, 30996]])]
)

conv_templates = {
    "vicuna": conv_vicuna_v1_1,
    "falcon": conv_falcon,
    "llama": conv_llama2,
    "chatgpt": conv_chatgpt,
    "gpt4": conv_chatgpt,
    "chatglm3": conv_chatglm3,
    "baichuan": conv_baichuan,
    "internlm": conv_internlm,
    "yiyan": conv_chatgpt,
    "xinghuo": conv_chatgpt,
    "qianwen": conv_chatgpt,
}

conv_templates_zh = {
    "llama": conv_llama2_zh,
    "baichuan": conv_baichuan_zh,
    "internlm": conv_internlm_zh,
    "chatgpt": conv_chatgpt_zh,
    "gpt4": conv_gpt4_zh,
    "yiyan": conv_chatgpt_zh,
    "xinghuo": conv_chatgpt_zh,
    "qianwen": conv_chatgpt_zh,
    "huatuo": conv_baichuan_zh,
    "chatglm3": conv_chatglm3_zh,
    "bloomz": conv_bloomz_zh,
}

prompt_templates = {
    "base_v1_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n",
    "base_v1_zh": "以下是一段医生和患者之间的对话。 为了更好地诊断患者，医生会轮流询问患者一些问题。 一旦医生认为他已经获得了有关患者的足够信息，他就会给出诊断。",

    "base_v2_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions according to the tips in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n\
**Tips:**:\n\
    1. The doctor's consultation generally follows the following oder:\n\
        a) age and gender\n\
        b) chief complaint and symptom\n\
        c) Basic information, including but not limited to, disease history, medication history, etc.\n\
        c) inspection results\n\
        d) laboratory results\n\
    2. The patient already has the examination and test results necessary for diagnosis, so when such results are needed, the doctor can directly ask the patient for the results rather than asking the patient to do.\n\n",
    "base_v2_zh": "以下是一段医生和患者之间网络问诊对话。为了更好地诊断患者，医生会轮流询问患者一些问题。该问诊对话的长度最多为10轮，医生需要获取患者的信息以确定患者的病因。一旦医生认为他已经获得了有关患者的足够信息，他就会给出诊断。\n",
    "ming_v2_zh": "[患者]: 你好，我最近有点不舒服。",

    "base_v3_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions according to the tips in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n\
**Tips:**:\n\
    1. The doctor's consultation generally follows the following oder:\n\
        a) Basic information, like age and gender, etc.\n\
        b) Chief complaint, like the primary symptom and duration, etc.\n\
        c) Cause of the symptom, like Travel history and past medical history, etc.\n\
        d) The results of the physican test, like the diagnostic imaging and vital signs.\n\
        e) Detail information, like disease history, medication history, etc.\n\
        f) More detail information that can help diagnosis \n\
    2. The patient has already undergone all the necessary examinations for diagnosis, and the doctor can directly inquire about the results of the tests without requiring the patient to undergo further examinations.\n\n",

    "base_v3_zh": "以下是一段医生和患者之间网络问诊对话。为了更好地诊断患者，医生会轮流询问患者一些问题，该问诊对话的长度最多为10轮。医生需要尽可能的获取患者的信息以确定患者的病因。一旦医生认为他已经获得了有关患者的足够信息，医生可以提前做出诊断。\n",
    "base_v3_en": "The following is an online medical consultation dialogue between a doctor and a patient. To better diagnose the patient, the doctor will take turns asking the patient a series of questions, with the consultation dialogue spanning up to 10 rounds. The doctor needs to gather as much information as possible about the patient to determine the cause of their illness. Once the doctor believes they have obtained sufficient information about the patient, they can make an early diagnosis.\n",

    "base_v4_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions according to the tips in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n\
**Tips:**:\n\
    1. The doctor's consultation generally follows the following oder:\n\
        a) Basic information, like age and gender, etc.\n\
        b) Chief complaint, like the primary symptom and duration, etc.\n\
        c) Cause of the symptom, like Travel history and past medical history, etc.\n\
        d) The results of the physican test, like the diagnostic imaging and vital signs.\n\
        e) More detail information that can help diagnosis \n\
    2. The patient has already undergone all the necessary examinations for diagnosis, so the doctor can directly inquire about the results of the tests without requiring the patient to do further examinations.\n\
    3. There are only a maximum of 10 rounds of consultation dialogue, so the questions asked by the doctor in each round should help determine the patient's disease as much as possible.\n\n",

    "base_v4_zh": "以下是一段医生和患者之间网络问诊对话。为了更好地诊断患者，医生根据<问诊提示>中的提示询问患者一些问题，该问诊对话的长度最多为10轮。医生需要尽可能的获取患者的信息以确定患者的病因。一旦医生认为他已经获得了有关患者的足够信息，医生可以提前做出诊断。\n\
<提示>：\n\
    医生一般可以按照以下顺序来对患者进行问诊：\n\
        1) 基本信息，包括但不限于年龄，性别，过往病史等。\n\
        2) 主诉，包括但不限于主要症状，发生时间等。\n\
        3) 病因，可能导致病人发生该症状的原因。\n\
        4) 检查及检验结果以便进一步确定诊断。\n\
        5) 给出诊断结果\n\n",


    "base_v5_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient a question according to the tips in each turn. Once the doctor feels that he has obtained enough information about the patient, he will give a diagnosis.\n\
**Tips:**:\n\
    1. The doctor's consultation generally follows the following order:\n\
        a) Basic information, like age and gender, etc.\n\
        b) Chief complaint, like the primary symptom and duration, etc.\n\
        c) Cause of the symptom, like Travel history and past medical history, etc.\n\
        d) The results of the physican test, like the diagnostic imaging and vital signs.\n\
    2. The doctor's questions should be concise and clear, while the tone should be patient and caring for the patient.\n\
    3. The patient has already undergone all the necessary examinations for diagnosis, so the doctor can directly inquire about the results of the tests without requiring the patient to do further examinations.\n\
    4. There are only a maximum of 10 rounds of consultation dialogue, so the questions asked by the doctor in each round should help to determine the patient's most likely diagnosis or to clarify the next medical examination that should be done as much as possible.\n\n",

}

def get_doctor_template(mode, model_name):
    model_name = model_name.lower()

    if mode == "medqa":
        return get_value(conv_templates, model_name)
    else:
        return get_value(conv_templates_zh, model_name)

def get_doctor_prompt(prompt_id):
    prompt_id = prompt_id.lower()

    # if model_name in conv_templates.keys():
    return prompt_templates[prompt_id]

