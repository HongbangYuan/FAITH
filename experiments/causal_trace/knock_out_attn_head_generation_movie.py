from core.inference.llama_inferencer import Llama2Inferencer

def untuple(x):
    return x[0] if isinstance(x, tuple) else x


if __name__ == '__main__':
    import os
    import argparse
    from dataset.ToyDataset.Movies.load_movies import load_wikidata_movies_knowing
    from utils import load_llama_model_and_tokenizer, get_model_name_mapping, CustomDataset, write_to_json, read_json
    from core.inference.llama_inferencer import format_question
    from tqdm import tqdm
    from utils.nethook import TraceDict,Trace
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, write_to_json, \
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
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=-1)
    args = parser.parse_args()

    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'

    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'

    model_size = str(args.model_size)+'b'
    model_name = "llama2-{}-chat".format(model_size)

    # load data
    result_file = f'{base_dir}/results/causal_trace/knockout/{model_name}_on_movies_knock_out.json'
    dataset_file = f'{base_dir}/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    orig_samples = read_json(dataset_file)
    # samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    # samples = [sample for sample in samples_one_time if
    #                     str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]
    samples = orig_samples
    acc = cal_acc_wiki_movies(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))

    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    generation_kwargs={
            "max_length":256,
            "num_beams":5,
            "do_sample":False,
    }
    print("generation kwargs:",generation_kwargs)

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
    print("Selected heads pos:",heads_pos)
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())

    pbar = tqdm(samples)
    for idx, sample in enumerate(pbar):
        question = sample[question_key]
        # question = f"Why was the film {sample['movie']} released in XXXX?"
        orig_prompts = [format_question(question,"")]
        batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
        batch_input = {
            key: value.to(model.device) for key, value in batch_input.items()
        }
        _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0],"".join(question.split()))
        pos = end_of_question - 2  # the previous token position before the question mark.

        def intervene_head(x,layer):
            h = untuple(x)
            if untuple(x).shape[1] == 1:
                # Pay close attention when prompt length=1! But I don't think this would happen.
                return x
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
            output = model.generate(
                input_ids=batch_input["input_ids"],
                attention_mask=batch_input["attention_mask"],
                **generation_kwargs,
            )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        answers = [
            remove_prompt(decoded_output[idx]) for idx in range(len(orig_prompts))
        ]
        sample["knockout_result"] = answers[0]
        write_to_json(samples[:idx], result_file, default=str)

    print("Writing result to {}!".format(result_file))
    print("Finished Running!")