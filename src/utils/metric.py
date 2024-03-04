import os
import copy

def get_item(_dict, key):
    copy_dict = copy.deepcopy(_dict)
    if isinstance(key, str):
        return copy_dict[key]
    elif isinstance(key, list):
        for k in key:
            # print(copy_dict)
            copy_dict = copy_dict[k]
            # print(copy_dict)
        return copy_dict
    else:
        raise NotImplementedError

def acc(datas, answer_key="answer", output_key="output"):
    acc_num = 0
    for data in datas:
        # print(data)
        reference = get_item(data, answer_key)
        prediction = get_item(data, output_key)
        if reference == prediction:
            acc_num += 1
    
    return acc_num / len(datas)