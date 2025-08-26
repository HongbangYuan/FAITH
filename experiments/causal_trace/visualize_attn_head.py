
from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch

def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def format_answer_from_sample(sample):
    # answer_template = "I apologize, but the film \"{}\" was not released in {}. The film was actually released in {}"
    # answer_template = "I apologize, but the film \"{}\" was actually released in {}"
    answer_template = "According to my knowledge, the film \"{}\" was released in {}"
    # answer_template = "According to my knowledge, the film you mentioned was released in {}"
    film_name = sample["movie"]
    false_year = sample["false_year"]
    true_year = str(sample["time"][0])
    # uncompleted_answer = answer_template.format(true_year[:-1])
    uncompleted_answer = answer_template.format(film_name, true_year[:-1])
    ground_truth_token = true_year[-1]
    return uncompleted_answer, ground_truth_token

if __name__ == '__main__':
    from utils.nethook import TraceDict,Trace
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices,select
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt,reshape_saliency_score
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from core.methods.causal_trace.casual_trace import find_token_range,decode_tokens
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution,get_distribution_from_hidden_state
    from collections import defaultdict
    import argparse
    from experiments.generation_attention.generation_attention_film import plot_attn_weight

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'

    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/visualize_attn_head'
    result_figs_dir = f'{result_dir}/figs'

    dataset_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer.json"
    samples = read_json(dataset_file)

    acc = cal_acc_wiki_movies(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers
    heads_pos = [(1,15),(2,2),(1,22),(5,15),(8,18)]
    layer_to_head = defaultdict(list)
    layernum_to_name = {layer:layername(model,layer,'self_attn') for layer in list(set(elem[0] for elem in heads_pos))}
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn')].append(elem[1])
    layers = list(layer_to_head.keys())

    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    curr_true_count = 0
    pbar = tqdm(samples)
    for idx, sample in enumerate(pbar):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        file_name = f'{result_dir}/{idx}_{sample_name}_attn_weight.npz'
        question = sample[question_key]

        if not os.path.isfile(file_name):
            print(f"Creating file {file_name}...")
            # question = f"Why was the film {sample['movie']} released in XXXX?"
            uncompleted_answer,ground_truth_token = format_answer_from_sample(sample)
            orig_prompt = format_question_answer(question, uncompleted_answer)
            batch_input = tokenizer([orig_prompt], return_tensors="pt", padding=True)
            batch_input = {
                key: value.to(model.device) for key, value in batch_input.items()
            }
            start_of_question, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0],"".join(question.split()))
            pos = end_of_question - 2  # the previous token position before the question mark.
            with torch.no_grad(),TraceDict(
                model,
                layers,
                retain_output=True
            ) as td:
                out = model(
                    **batch_input,
                    output_hidden_states=True,
                    output_attentions=True,
                )
            states = []
            for layer,head in heads_pos:
                layer_name = layernum_to_name[layer]
                states.append(td[layer_name].output[1][:,head,:,:].squeeze().cpu().numpy())
            attn_weights = np.stack(states)
            # print("Debug Usage")

            question_range = find_token_range(tokenizer,batch_input["input_ids"][0],"".join(question.split()))
            start,end = question_range

            token_attn_weight = attn_weights[:,start:end,start:end]
            token_list = decode_tokens(tokenizer,batch_input["input_ids"][0][start:end])

            result = {
                "attn_weights":attn_weights,
                "token_attn_weight":token_attn_weight,
                "token_list":token_list
            }
            np.savez(file_name,**result)
        else:
            result = np.load(file_name,allow_pickle=True)

        token_attn_weight = result["token_attn_weight"]
        token_list = result["token_list"]

        for i,head_pos in enumerate(heads_pos):
            head_name = "-".join(map(str, head_pos))
            pdf_fig_name = f'{idx}_{sample_name}_{head_name}.pdf'
            pdf_save_file = f"{result_figs_dir}/{pdf_fig_name}"
            flag = "True" if sample[token_result_key] else "False"
            pdf_title = f"{idx}_{sample_name}_{head_name}_{flag}"
            plot_attn_weight(token_attn_weight[i],word_list=token_list,title=pdf_title,save_path=pdf_save_file)


