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

    def get_prompt(self, patient_info):
        if self.sep_style == SeparatorStyle.VICUNA:
            seps = [self.sep, self.sep2]
            ret = self.system.format(patient_info=patient_info) + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message :
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.FALCON:
            ret = self.system.format(patient_info=patient_info) + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "

            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = self.system.format(patient_info=patient_info)
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
            ret = self.system.format(patient_info=patient_info)
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ": " + message + seps[i % 2] + "\n"
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = self.system.format(patient_info=patient_info)
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
    stop_ids=[torch.tensor([3970, 1783, 1955]), torch.tensor([11662, 1783, 1955]), torch.tensor([29871, 13, 29925, 1299, 29902, 3919]), torch.tensor([349, 1299, 29902, 3919])]
)

conv_chatgpt = Conversation(
    system="{prompt}",
    roles=("DOCTOR", "PATIENT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.VICUNA,
    sep="\n",
    sep2="\n",
    stop_ids=[]
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

conv_internlm_zh = Conversation(
    system="{prompt}",
    roles=("[医生]", "[患者]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_ids=[torch.tensor([103028]), torch.tensor([336, 68305, 332]), torch.tensor([336, 68049, 332])]
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
}

conv_templates_zh = {
    "llama": conv_llama2_zh,
    "baichuan": conv_baichuan_zh,
    "internlm": conv_internlm_zh,
    "chatgpt": conv_chatgpt_zh,
    "gpt4": conv_chatgpt_zh,
    "yiyan": conv_chatgpt_zh,
    "xinghuo": conv_chatgpt_zh,
    "qianwen": conv_chatgpt_zh,
    "chatglm3": conv_chatglm3_zh,
    "ming": conv_chatgpt_zh,
}

prompt_templates = {
    "base_v1_en": "A chat between a skilled doctor and a patient in need. In order to better diagnose the patient, the doctor will ask the patient some questions, and the patient will answer according to his or her physical condition. The patient will only answer the information asked by the doctor and will not disclose any additional information that the doctor has not mentioned. \n**Patient Physical Condition**: {patient_info}\n",
    
    "base_v1_en_new": "Below is a chat between a doctor and a patient. In order to better diagnose the patient, the doctor will ask the patient some questions, and the patient will answer according to his or her **Patient Physical Condition** with the **Requirements**.\n\n\
**Requirements**: \n\
    1. The patient answer the information asked by the doctor according to the patient physical condition. \n\
    2. The patient should not disclose any additional information that the doctor has not mentioned and required.\n\
**Patient Physical Condition**: {patient_info}\n\n",
    
    "base_v1_zh": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据自身的身体情况对医生的问题进行回复。患者只会回答与医生问题相关的身体情况，不会透露任何与医生问题无关的额外身体情况信息，\n**患者身体情况信息**: {patient_info}\n",
    
    "base_v2_en": "Below is a chat between a doctor and a patient. In order to better diagnose the patient, the doctor will ask the patient some questions, and the patient will answer according to his or her **Patient Physical Condition** with the **Requirements**.\n\n\
**Requirements**: \n\
    1. The patient answer the information asked by the doctor according to the patient physical condition. \n\
    2. The patient should not disclose any additional information that the doctor has not mentioned and required.\n\
    3. The Patients should not make up information that does not exist in the patient physical condition. If the information asked does not exist in the patient physical condition, the patient should reply with {{Sorry, I don't know such information.}}\n\
    4. When answering each question, the patient can only answer one key point in the patient's information at most. For some questions that are too broad, patients can ask the doctor to ask more specific.\n\
    5. The patient's purpose is to ask the doctor for help. He should refuse to answer any questions that are not related to the consultation and continue to express his needs to the doctor with {{Sorry, I don't want to talk about this right now. What else do you want to know about my situation?}}\n\n\
**Patient Physical Condition**: {patient_info}\n\n",

    "base_v2_zh": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者应该回复{{抱歉，我不知道这个信息。}}\n\
    4. 在回答医生信息时，患者应该尽可能少的透露自身的信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\n\
**患者身体情况**: {patient_info}\n\n",

    "base_v3_en": "Below is a chat between a doctor and a patient. In order to better diagnose the patient, the doctor will ask the patient some questions, and the patient will answer according to his or her **Patient Physical Condition** with the **Requirements**.\n\n\
**Requirements**: \n\
    1. The patient answer the information asked by the doctor according to the patient physical condition. \n\
    2. The patient should not disclose any additional information that the doctor has not mentioned and required.\n\
    3. The Patients should not make up information that does not exist in the patient physical condition. If the information asked does not exist in the patient physical condition, the patient should reply with {{Sorry, I don't know such information.}}\n\
    4. When answering each question, the patient can only answer one key point in the patient's information at most. For some questions that are too broad, patients should ask the doctor to ask more specific.\n\
    5. The patient's purpose is to ask the doctor for help. He should refuse to answer any questions that are not related to the consultation and continue to express his needs to the doctor with {{Sorry, I don't want to talk about this right now. What else do you want to know about my situation?}}\n\
    6. If the doctor requires the patient to undergo a test that is included in the patient information, the patient should directly tell the doctor the results of the test.\n\n\
**Patient Physical Condition**: {patient_info}\n\n",

    "base_v6_zh": """
<回复要求>: 医生的问题分为一下8种类型，患者需要自行判断医生问题的类型，并按照对应的下述要求对医生的问题进行回复。
    询问信息类:
        1. 医生的问题很较为具体，且所问信息在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该问题部分的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        2. 医生的问题很较为具体，但所问信息不在<患者身体情况>中。要求：患者可以默认该情况不存在，应该对这个问题予以否定。
        3. 医生的问题较为宽泛。要求：患者应该要求医生问的再具体一点。
    医疗建议类:
        4. 医生的建议很较为具体，且建议的结果在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该建议结果的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        5. 医生的建议很较为具体，但建议的结果不在<患者身体情况>中。要求：患者应该表示没尝试过该建议，并表示会听从该建议。
        6. 医生的建议较为宽泛。要求：患者应该要求医生说明具体的建议是什么。
    物理动作类:
        7. 医生要求患者做出一些物理动作以配合治疗。要求：患者应该拒绝，并提醒医生这是网络对话。
    其他话题类:
        8. 医生的问题和建议与问诊无关。要求：患者应该要求医生回到问诊情景下。

<患者身体情况>: {patient_info}

以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据<回复要求>和<患者身体情况>对医生的问题进行回复。

""",

    "base_v7_zh": """
<回复要求>: 医生的问题分为一下8种类型，患者需要自行判断医生问题的类型，并按照对应的下述要求对医生的问题进行回复。
    询问信息类:
        1. 医生的问题很较为具体，且所问信息在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该问题部分的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        2. 医生的问题很较为具体，但所问信息不在<患者身体情况>中。要求：患者可以默认该情况不存在，应该对这个问题予以否定。
        3. 医生的问题较为宽泛。要求：患者应该要求医生问的再具体一点。
    医疗建议类:
        4. 医生的建议很较为具体，且建议的结果在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该建议结果的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        5. 医生的建议很较为具体，但建议的结果不在<患者身体情况>中。要求：患者应该表示没尝试过该建议，并表示会听从该建议。
        6. 医生的建议较为宽泛。要求：患者应该要求医生说明具体的建议是什么。
    物理动作类:
        7. 医生要求患者做出一些物理动作以配合治疗。要求：患者应该拒绝，并提醒医生这是网络对话。
    其他话题类:
        8. 医生的问题和建议与问诊无关。要求：患者应该要求医生回到问诊情景下。

<患者身体情况>: {patient_info}

以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据<回复要求>和<患者身体情况>对医生的问题进行回复。注意：直接以患者的语气以第一人称的形式回复即可，不要输出任何<回复要求>中的文本内容！

""",

    "base_v8_zh": """
<回复要求>: 医生的问题分为一下8种类型，患者需要自行判断医生问题的类型，并按照对应的下述要求对医生的问题进行回复。
    询问信息类:
        1. 医生的问题很较为具体，且所问信息在<患者身体情况>中。要求：患者应该尽量用<患者身体情况>中有关该问题部分的文本回答医生的问题，保持患者信息的的准确性。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        2. 医生的问题很较为具体，但所问信息不在<患者身体情况>中。要求：患者可以默认该情况不存在，应该对这个问题予以否定。
        3. 医生的问题较为宽泛。要求：患者应该要求医生问的再具体一点。
    医疗建议类:
        4. 医生的建议很较为具体，且建议的结果在<患者身体情况>中。要求：患者应该尽量用<患者身体情况>中有关该建议结果的文本回答医生的问题，保持患者信息的的准确性。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
        5. 医生的建议很较为具体，但建议的结果不在<患者身体情况>中。要求：患者应该表示没尝试过该建议，并表示会听从该建议。
        6. 医生的建议较为宽泛。要求：患者应该要求医生说明具体的建议是什么。
    物理动作类:
        7. 医生要求患者做出一些物理动作以配合治疗。要求：患者应该拒绝，并提醒医生这是网络对话。
    其他话题类:
        8. 医生的问题和建议与问诊无关。要求：患者应该要求医生回到问诊情景下。

<患者身体情况>: {patient_info}

以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据<回复要求>和<患者身体情况>对医生的问题进行回复。注意：直接以患者的语气以第一人称的形式回复即可，不要输出任何<回复要求>中的文本内容！

""",

}

state_prompt_templates = {
    "base_v1_zh": {
        "A": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会**患者身体情况信息**对医生的问题进行回复。患者只会回答与医生问题相关的身体情况，不会透露任何与医生问题无关的额外身体情况信息，\n**患者身体情况信息**: {patient_info}\n",
        "B": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 若医生的问题中有具体的医疗检查类型，如果**患者身体情况信息**中有该检查的结果，患者会直接回复该结果。如果没有该检查的结果，患者会回复该检查没有出现异常情况。\n\
    2. 若医生的问题中没有具体的医疗检查类型，此时患者应该询问医生具体是什么检查。\n\
    3. 注意，不要询问医生检查的具体做法\n\n\
**患者身体情况信息**：{patient_info}\n\n",
        "C": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 若医生的问题中有具有的外科检查目的或者外科检查动作要求，如果**患者身体情况信息**中有该外科检查的结果，患者会直接回复该结果。如果没有，那么患者会回复医生该物理检查没有出现异常情况。\n\
    2. 若医生的问题中不含有任何具体的外科检查目的和外科检查动作要求，那么患者会询问医生具体应该怎么配合，或者询问该外科检查的目的。\n\n\
**患者身体情况信息**： {patient_info}\n\n",
        "D": None,
    },


    "base_v2_zh": {
        "A": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者应该予以否定。\n\
    4. 在回答医生信息时，患者应该尽可能少的透露自身的信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\n\
**患者身体情况**: {patient_info}\n\n",

        "B": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 医生的问题中含有具体的医疗检查类型时，如果**患者身体情况**中有该检查的结果，患者会直接回复该结果。\n\
    2. 医生的问题中含有具体的医疗检查类型时，如果**患者身体情况**中没有该检查的结果，患者会回复没做过该检查。\n\
    3. 医生的问题中不含有医疗检查类型或医疗检查类型较为模糊时，患者应该询问医生具体是什么检查。\n\
    4. 注意，患者不要让医生解释检查的目的和具体做法，应该直接回答医生的问题。\n\n\
**患者身体情况**：{patient_info}\n\n",

        "C": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 若医生的问题中有具有的外科检查目的或者外科检查动作要求，如果**患者身体情况信息**中有该外科检查的结果，患者会直接回复该结果。如果没有，那么患者会回复医生该物理检查没有出现异常情况。\n\
    2. 若医生的问题中不含有任何具体的外科检查目的和外科检查动作要求，那么患者会询问医生具体应该怎么配合，或者询问该外科检查的目的。\n\n\
**患者身体情况信息**： {patient_info}\n\n",

        "D": None,
    },

    "base_v3_zh": {
        "A": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者应该予以否定。\n\
    4. 在回答医生信息时，患者应该尽可能少的透露自身的信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\n\
**患者身体情况**: {patient_info}\n\n",

        "B": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 具体的医疗检查检查项目包括但不限于CT，MRI，X光，活体细胞检查，肠胃镜检查等。如果只是简单的提到检查和治疗，则不算具体。\n\
    2. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中有该检查的结果，患者会直接回复该结果。\n\
    3. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中没有该检查的结果，患者会回复没做过该检查。\n\
    4. 医生的问题中不含有检查项目或检查类型较为模糊时，患者应该询问医生具体是什么检查。\n\
    5. 注意，患者不要反问医生检查的具体目的和做法。\n\n\
**患者身体情况**：{patient_info}\n\n",

        "C": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 若医生的问题中有具有的外科检查目的或者外科检查动作要求，如果**患者身体情况信息**中有该外科检查的结果，患者会直接回复该结果。如果没有，那么患者会回复医生该物理检查没有出现异常情况。\n\
    2. 若医生的问题中不含有任何具体的外科检查目的和外科检查动作要求，那么患者会询问医生具体应该怎么配合，或者询问该外科检查的目的。\n\n\
**患者身体情况信息**： {patient_info}\n\n",

        "D": None,
    },

    "base_v4_zh": {
        "A": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题，回答的信息一定要和医生的问题是对应的。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者可以默认该症状或者情况不存在。\n\
    4. 在回答医生信息时，患者应该尽可能少的描述信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\n\
**患者身体情况**: {patient_info}\n\n",

        "B": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 具体的医疗检查检查项目包括但不限于CT，MRI，X光，活体细胞检查，肠胃镜检查等。如果只是简单的提到检查和治疗，则不算具体。\n\
    2. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中有该检查的结果，患者会直接回复该结果。\n\
    3. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中没有该检查的结果，患者会回复没做过该检查。\n\
    4. 医生的问题中不含有检查项目或检查类型较为模糊时，患者应该询问医生具体是什么检查。\n\
    5. 注意，患者不要反问医生检查的具体目的和做法。\n\n\
**患者身体情况**：{patient_info}\n\n",

        "C": "以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n\
**回复要求**：\n\
    1. 物理动作包括需要患者作出物理意义上的行为和行动，包括但不限于举手，脱衣服，站直，躺下，去医院就医等。\n\
    2. 医生的问题中要求患者做出具体的物理动作时，患者应该表示这是网络问诊，无法立刻完成此类动作，并希望医生能够继续询问其他信息或者依靠已有的信息做出诊断。\
**患者身体情况信息**： {patient_info}\n\n",

        "D": None,
    
    },

    "base_v5_zh": {
        "A": "\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者可以默认该症状或者情况不存在。\n\
    4. 在回答医生信息时，患者应该尽可能少的描述信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\
    5. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况**: {patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n",

        "B": "\
**回复要求**：\n\
    1. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中有该检查的结果，患者会直接回复该结果。\n\
    2. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中没有该检查的结果，患者会回复没做过该检查。\n\
    3. 医生的问题中不含有检查项目或检查类型较为模糊时，患者应该询问医生具体是什么检查。\n\
    4. 注意，患者不要反问医生检查的具体目的和做法。\n\
    5. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况**：{patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n",

        "C": "\
**回复要求**：\n\
    1. 物理动作包括需要患者作出物理意义上的行为和行动，包括但不限于举手，脱衣服，站直，躺下，去医院就医等。\n\
    2. 医生的问题中要求患者做出具体的物理动作时，患者应该表示这是网络问诊，无法立刻完成此类动作，并希望医生能够继续询问其他信息或者依靠已有的信息做出诊断。\n\n\
    3. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况信息**： {patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**对医生的问题进行回复。\n",

        "D": None,
    },

    "base_v6_zh": {
        "A": "\
**回复要求**：\n\
    1. 患者根据**患者身体情况**中的信息来回复医生的问题。\n\
    2. 患者不会透露任何医生没有提到或者询问的信息。\n\
    3. 患者不能编造**患者身体情况**中没有的信息，如果医生所询问的信息不存在于**患者身体情况**中，那么患者可以默认该症状或者情况不存在。\n\
    4. 在回答医生信息时，患者应该尽可能少的描述信息。对于医生过于宽泛的问题，病人可以要求医生询问的具体一些。\n\
    5. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况**: {patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**以第一人称的形式对医生的问题进行回复。\n",

        "B": "\
**回复要求**：\n\
    1. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中有该检查的结果，患者会直接回复该结果。\n\
    2. 医生的问题中含有具体的检查项目时，如果**患者身体情况**中没有该检查的结果，患者会回复没做过该检查。\n\
    3. 医生的问题中不含有检查项目或检查类型较为模糊时，患者应该询问医生具体是什么检查。\n\
    4. 注意，患者不要反问医生检查的具体目的和做法。\n\
    5. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况**：{patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**以第一人称的形式对医生的问题进行回复。\n",

        "C": "\
**回复要求**：\n\
    1. 物理动作包括需要患者作出物理意义上的行为和行动，包括但不限于举手，脱衣服，站直，躺下，去医院就医等。\n\
    2. 医生的问题中要求患者做出具体的物理动作时，患者应该表示这是网络问诊，无法立刻完成此类动作，并希望医生能够继续询问其他信息或者依靠已有的信息做出诊断。\n\n\
    3. 注意，患者的回复不要出现上文已经回答过的医疗信息。\n\n\
**患者身体情况信息**： {patient_info}\n\n\
以下是一段医生和患者之间的对话。为了能够更好的诊断患者，医生会询问患者几个问题，而患者则会根据**回复要求**以第一人称的形式对医生的问题进行回复。\n",

        "D": None,
    },

    "base_v8_zh": {
        "A-A-A": "<患者身体情况>: {patient_info}\n\
<当前回复要求>: 请用<患者身体情况>中的所有原文回复医生的问题，注意一定要用<患者身体情况>中的原文来回答，从而保持患者信息的的准确性。\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "A-A-B": "<当前回复要求>: 患者没有医生所询问的症状，请对当前医生的问题予以否定。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "A-B":  "<当前回复要求>: 当前医生的问题太过宽泛，患者会要求医生的问题更加具体一些。注意！不要回复医生的问题！{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "B-A-A": "<患者身体情况>: {patient_info}\n\
<当前回复要求>: 医生当前的建议患者已经采用过，请用<患者身体情况>中的所有原文回复医生的建议，不要改变其说法，保持患者信息的的准确性。\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "B-A-B": "<当前回复要求>: 患者没有尝试过医生当前的建议，可以表示会听从医生的建议。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "B-B":  "<当前回复要求>: 当前医生的建议太过宽泛，患者会要求医生的建议更加具体一些。注意，患者应该要求医生给出具体的检查项目或者治疗方案，而不是要求医生给出检查和治疗的细节。注意！不要回复医生的问题！{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "C": "<当前回复要求>: 提醒医生当前是网络问诊，无法完成对应的物理动作。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
        "D": "<当前回复要求>: 提醒医生偏离了问诊主题，要求其回到问诊情景下。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！",
    },

    "base_v9_zh": {
        "A-A-A": "<患者身体情况>: {patient_info}\n\
<当前回复要求>: 请用<患者身体情况>中的所有原文回复医生的问题，注意一定要用<患者身体情况>中的原文来回答，从而保持患者信息的的准确性。\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "A-A-B": "<当前回复要求>: 患者没有医生所询问的症状，请对当前医生的问题予以否定。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "A-B":  "<当前回复要求>: 医生当前的问题太过宽泛，患者会要求医生就当前最新一轮的问题问得更加具体一些。注意，不要编造任何不存在的信息，也不要询问医生的问题。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "B-A-A": "<患者身体情况>: {patient_info}\n\
<当前回复要求>: 医生当前的建议患者已经采用过，请用<患者身体情况>中的所有原文回复医生的建议，不要改变其说法，保持患者信息的的准确性。\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "B-A-B": "<当前回复要求>: 患者没有尝试过医生当前的建议，可以表示会听从医生的建议。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "B-B":  "<当前回复要求>: 当前医生的建议太过宽泛，患者会要求医生就当前最新一轮的建议更加具体一些。注意，不要编造任何不存在的信息，也不要询问医生问题。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "C": "<当前回复要求>: 提醒医生当前是网络问诊，无法完成对应的物理动作。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
        "D": "<当前回复要求>: 提醒医生偏离了问诊主题，要求其回到问诊情景下。{patient_info}\n\
以下是一段医生和患者之间的对话。患者则会根据<当前回复要求>以第一人称的形式对当前医生最新一轮的问题进行回复。注意，不要输出任何<当前回复要求>中的文本内容！\n",
    },

    "base_v9_en": {
        "A-A-A": "<Patient Condition>: {patient_info}\n\
<Response Requirement>: Please respond to the doctor's question using all the original text from <Patient Condition> to ensure the accuracy of the patient information.\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "A-A-B": "<Response Requirement>: The patient does not have the symptoms inquired by the doctor, please deny the current doctor's question. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "A-B": "<Response Requirement>: The doctor's current question is too broad, and the patient will ask the doctor to be more specific about the latest round of questions. Note, do not make up any non-existent information, nor ask questions to the doctor. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "B-A-A": "<Patient Condition>: {patient_info}\n\
<Response Requirement>: The doctor's current suggestion has already been tried by the patient, please reply to the doctor's advice using all the original text from <Patient Condition>, without changing its statement, to maintain the accuracy of the patient information.\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "B-A-B": "<Response Requirement>: The patient has not tried the doctor's current suggestion and can express willingness to follow the doctor's advice. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "B-B": "<Response Requirement>: The doctor's current advice is too broad, and the patient will request the doctor to be more specific about the latest round of advice. Note, do not make up any non-existent information, nor ask questions to the doctor. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "C": "<Response Requirement>: Remind the doctor that this is an online consultation, and it is not possible to perform the corresponding physical actions. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
        "D": "<Response Requirement>: Remind the doctor that they have deviated from the consultation topic, and request to return to the consultation scenario. {patient_info}\n\
Below is a dialogue between a doctor and a patient. The patient will respond to the latest round of the doctor's question in the first person, based on <Response Requirement>. Note, do not output any text content from <Response Requirement>!\n",
}
}

# <回复要求>: 医生的问题分为一下8种类型，患者需要自行判断医生问题的类型，并按照对应的下述要求对医生的问题进行回复。
#     询问信息类:
#         1. 医生的问题很较为具体，且所问信息在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该问题部分的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
#         2. 医生的问题很较为具体，但所问信息不在<患者身体情况>中。要求：患者可以默认该情况不存在，应该对这个问题予以否定。
#         3. 医生的问题较为宽泛。要求：患者应该要求医生问的再具体一点。
#     医疗建议类:
#         4. 医生的建议很较为具体，且建议的结果在<患者身体情况>中。要求：患者应该复制<患者身体情况>中有关该建议结果的文本回答医生的问题。注意，患者应该只回复和医生问题相关的信息，不要回复无关的内容以及透露无关的信息。
#         5. 医生的建议很较为具体，但建议的结果不在<患者身体情况>中。要求：患者应该表示没尝试过该建议，并表示会听从该建议。
#         6. 医生的建议较为宽泛。要求：患者应该要求医生说明具体的建议是什么。
#     物理动作类:
#         7. 医生要求患者做出一些物理动作以配合治疗。要求：患者应该拒绝，并提醒医生这是网络对话。
#     其他话题类:
#         8. 医生的问题和建议与问诊无关。要求：患者应该要求医生回到问诊情景下。


def get_patient_template(mode, model_name):
    model_name = model_name.lower()

    if mode == "medqa":
        return get_value(conv_templates, model_name)
    else:
        return get_value(conv_templates_zh, model_name)

def get_patient_prompt(prompt_id, state=None):
    prompt_id = prompt_id.lower()

    if state is not None:
        return state_prompt_templates[prompt_id][state]
    return prompt_templates[prompt_id]

