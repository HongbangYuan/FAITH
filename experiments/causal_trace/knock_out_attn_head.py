from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch
import random

def untuple(x):
    return x[0] if isinstance(x, tuple) else x

def generate_random_positions(num_positions, model_size, seed=None):
    random.seed(seed)
    positions = []
    for _ in range(num_positions):
        x = random.randint(0, 8)
        y = random.randint(0, 39)
        positions.append((x, y))
    return positions


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
    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices,select
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from core.methods.causal_trace.casual_trace import find_token_range
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution,get_distribution_from_hidden_state
    from collections import defaultdict
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker',action='store_true')
    args = parser.parse_args()

    use_docker = args.use_docker
    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    dataset_file = f'{base_dir}/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    orig_samples = read_json(dataset_file)
    samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    samples = [sample for sample in samples_one_time if
                        str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]

    acc = cal_acc_wiki_movies(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers
    if model_size == '7b':
        heads_pos = [(1,15),(2,2),(1,22),(5,15),(8,18)] # Movie attn head 7b
        # heads_pos = [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)]  # Nobel Prize attn head7b
    else:
        heads_pos = [(8, 14), (2, 31), (10, 11), (12, 38), (2, 7)] # Movie attn 13b
        # heads_pos = [(0, 13), (18, 2), (2, 31), (1, 28), (15, 22)]  # Nobel Prize attn head13b
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())

    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    curr_true_count = 0
    pbar = tqdm(samples)
    print(f"Processing a total of {len(samples)} samples...")
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            question = sample[question_key]
            # question = f"Why was the film {sample['movie']} released in XXXX?"
            uncompleted_answer,ground_truth_token = format_answer_from_sample(sample)
            orig_prompt = format_question_answer(question, uncompleted_answer)
            batch_input = tokenizer([orig_prompt], return_tensors="pt", padding=True)
            batch_input = {
                key: value.to(model.device) for key, value in batch_input.items()
            }
            _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0],"".join(question.split()))
            pos = end_of_question - 2  # the previous token position before the question mark.

            def intervene_head(x,layer):
                h = untuple(x)
                heads = layer_to_head[layer]
                for head in heads:
                    dim_start = head * head_dim
                    dim_end = (head+1) * head_dim
                    # h[:,pos,dim_start:dim_end] = torch.zeros_like(h[:,pos,dim_start:dim_end])
                    h[:,pos,dim_start:dim_end] = 0
                return x

            with torch.no_grad(),TraceDict(
                model,
                layers,
                edit_input=intervene_head
            ):
                out = model(
                    **batch_input,
                    output_hidden_states=True,
                )
                probs = torch.softmax(out["logits"][:, -1], dim=1)
                base_score, answer = torch.max(probs, dim=1)
                predicted_token = tokenizer.decode(answer)
            sample["ground_truth_token"] = ground_truth_token
            sample[token_result_key] = (ground_truth_token == predicted_token,predicted_token,base_score.item())
            curr_true_count += sample[token_result_key][0]
            pbar.set_description(f"{curr_true_count}/{idx+1}")
            results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])

    token_predictions = [sample[token_result_key][0] for sample in samples]
    token_acc = sum(token_predictions) / len(results)
    print("token_acc:",token_acc)
    analyze_samples = [sample if sample[token_result_key][0] == False else None for sample in samples]
    # analyze_samples = [sample for sample in samples  if sample[token_result_key][0] == False]
    # write_to_json(analyze_samples,f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer_false.json")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)


