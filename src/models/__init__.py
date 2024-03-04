from .base_model import *
from .openai_model import *
from .qianwen_model import *
from .xinghuo_model import *
from .yiyan_model import *

def get_model(model_name, stop_ids=[]):
    # api model
    if model_name == "chatgpt":
        return OpenAI_Model(model_type="gpt-3.5-turbo-1106", stop_ids=stop_ids)
    elif model_name == "gpt4":
        return OpenAI_Model(model_type="gpt-4-1106-preview", stop_ids=stop_ids)
    elif model_name == "qianwen":
        return QianWen_Model(stop_ids=stop_ids)
    elif model_name == "xinghuo":
        return XingHuo_Model(stop_ids=stop_ids)
    elif model_name == "yiyan":
        return YiYan_Model(stop_ids=stop_ids)
    else:
        return Local_Model(model_name, stop_ids)
    