import matplotlib.pyplot as plt
import nltk
import string
import os
from baukit import TraceDict

llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
llama_prompt_template = llama_sys_prompt + ' {} [/INST] {}'


def format_question_answer(question,answer):
    return llama_prompt_template.format(question,answer)

def remove_subtring(text,signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal,text))
    return text[end_idx:].strip()

def remove_prompt(text):
    return remove_subtring(text,'[/INST]')



if __name__ == '__main__':
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    from tqdm import tqdm
    import json
    import numpy as np
    from utils import read_json,select
    import argparse
    from core.methods.iti.iti import get_llama_activations_bau
    from core.evaluation.film_release_evaluation import cal_acc

    load_dotenv()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from dataset.ToyDataset.Movies.load_movies import load_movies,load_film_release
    from utils import select, CustomDataset,write_to_json,read_json,model_name_mapping,load_llama_model_and_tokenizer

    # select some arguments
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size',default=8,help='batch_size')


    # Parse the arguments
    args = parser.parse_args()

    # select some arguments
    batch_size = args.batch_size
    model_size = str(args.model_size)+'b'
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_dataset_get_activation.json'.format(model_name)
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_dataset_untruthful_instruction.json'.format(model_name)
    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/attention/iti'
    print("Model Name:",model_name_or_path)
    print("Result Path:",result_path)

    data_source = "/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_dataset.json".format(model_name)
    film_release_dataset = read_json(data_source)
    if model_size == '7b':
        key = 'false_premise_question_answer'
    elif model_size == '13b':
        key = 'false_premise_model_answer'
    else:
        raise ValueError("Model size {} not supported!".format(model_size))
    acc = cal_acc(film_release_dataset,key=key)
    print("data_source:",data_source)
    print("acc:",acc)

    model,tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    for sample in tqdm(film_release_dataset):
        prompt = format_question_answer(sample["false_premise_question"],sample[key])
        encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids
        layer_wise_activations, head_wise_activations = get_llama_activations_bau(model, encoded_prompt)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        all_head_wise_activations.append(head_wise_activations[:,-1,:])

    layer_wise_activation_file = f'{result_path}/{model_name}_feature_layer_wise.npy'
    print(f"Saving layer wise activations to {layer_wise_activation_file}")
    np.save(layer_wise_activation_file, all_layer_wise_activations)

    head_wise_activation_file = f'{result_path}/{model_name}_feature_head_wise.npy'
    print(f"Saving head wise activations to {head_wise_activation_file}")
    np.save(head_wise_activation_file, all_head_wise_activations)

    label_file = f'{result_path}/{model_name}_labels.npy'
    print(f"Saving labels to {label_file}")
    labels = [sample["pred"] for sample in film_release_dataset]
    np.save(label_file,labels)




