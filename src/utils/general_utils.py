import os
import pdb
import json
import numpy as np

def npy2json(npy_path):
    if os.path.exists(npy_path):
        seed_tasks = np.load(npy_path, allow_pickle=True).tolist()
        return seed_tasks
    else:
        raise FileNotFoundError

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def hisotry2str(historys, first_key="doctor", second_key="patient"):
    history_str = ""
    for history in historys:
        history_str += f"{first_key.upper()}: {history[first_key]}\n"
        if second_key in history.keys():
            # assert history == historys[-1], f"{historys}\nCONVERSATION MISS ERROR!"
            history_str += f"{second_key.upper()}: {history[second_key]}\n"
    
    return history_str

def mkdir(file_path):
    folder_path = file_path.rsplit('/', 1)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_value(_dict, key):
    if key in _dict.keys():
        return _dict[key]
    else:
        for k in _dict.keys():
            if k in key:
                return _dict[k]

def chunk_split(datas, chunk_size=50):
    chunk_splited_datas = []

    chunk_id = 0
    while True:
        if len(datas) >= chunk_size * (chunk_id+1):
            chunk_splited_datas.append(datas[chunk_size * chunk_id:chunk_size * (chunk_id+1)])
            chunk_id += 1
        else:
            chunk_splited_datas.append(datas[chunk_size * chunk_id:])
            return chunk_splited_datas, chunk_id+1