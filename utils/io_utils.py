import os
import random
import json
from tqdm import tqdm
import numpy as np
import yaml
import pickle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_jsonl(file):
    with open(file, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def write_to_jsonl(samples, file):
    with open(file, "w") as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')


def load_yaml(file_name):
    with open(file_name) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def read_multiple_jsonl(file_name_list):
    results = []
    for file_name in file_name_list:
        samples = read_jsonl(file_name)
        results.extend(samples)
    return results


def read_multiple_json(file_name_list):
    results = []
    for file_name in file_name_list:
        samples = read_json(file_name)
        results.extend(samples)
    return results


def write_to_json(samples, file,**kwargs):
    with open(file, "w") as f:
        json.dump(samples, f,**kwargs)


def merge_multiple_json(file_name_list,file):
    samples = []
    for file_name in file_name_list:
        sub_samples = read_json(file_name)
        samples.extend(sub_samples)
    write_to_json(samples,file)

def write_to_pickle(elements, file):
    with open(file, 'wb') as handle:
        pickle.dump(elements, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file):
    with open(file, "rb") as handle:
        return pickle.load(handle)