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


def get_llama_activations_bau(model, prompt):

    model.eval()

    HEADS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
    # MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    MLPS = []

    with torch.no_grad():
        prompt = prompt.to(model.device)
        with TraceDict(model, HEADS+MLPS, retain_output=True, retain_input=True, clone=True, detach=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output[0].squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        # mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        # mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states



if __name__ == '__main__':
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    from tqdm import tqdm
    import json
    import numpy as np
    from utils import read_json,select
    import argparse
    from core.evaluation.film_release_evaluation import cal_acc
    from core.evaluation.noble_prize.nobel_prize_when_evaluation import cal_when_acc

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
    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/activation/nobel_prize'
    print("Model Name:",model_name_or_path)
    print("Result Path:",result_path)

    data_source = "/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{}-chat_on_film_noble_prize_when_question.json".format(model_size)
    samples = read_json(data_source)
    question_key = 'when_question'
    answer_key = 'when_answer'
    acc = cal_when_acc(samples,pred_key=answer_key,ground_truth_key='when_answer_ground_truth',result_key='pred')
    print("data_source:",data_source)
    print("acc:",acc)

    model,tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    for sample in tqdm(samples):
        prompt = format_question_answer(sample[question_key],sample[answer_key])
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
    labels = [sample["pred"] for sample in samples]
    np.save(label_file,labels)




