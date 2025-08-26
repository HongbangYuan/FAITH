from core.methods.causal_trace.casual_trace import find_token_range
from core.methods.causal_trace.causal_trace_tok_pred import predict_from_input,decode_tokens
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def untuple(x):
    return x[0] if isinstance(x, tuple) else x

def calculate_hidden_flow(
        model,
        tokenizer,
        question_reference,
        question_counter_factual,
        uncompleted_answer,
        expect=None,
        kind=None,
        window=5
):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_layers = model.config.num_hidden_layers

    orig_prompts = [
        format_question_answer(question_reference, uncompleted_answer),
        format_question_answer(question_counter_factual, uncompleted_answer),
        format_question_answer(question_reference, uncompleted_answer),
    ]
    batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
    batch_input = {
        key:value.to(model.device) for key,value in batch_input.items()
    }
    with torch.no_grad():
        answers_t, base_scores = [[d[0],d[1]] for d in predict_from_input(model, batch_input)]
    [predicted_token,counter_factual_token] = decode_tokens(tokenizer, answers_t)
    if predicted_token == ground_truth_token or counter_factual_token != ground_truth_token:
        return None

    _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question_reference.split()))
    answer_start_pos, _ = find_token_range(tokenizer,batch_input["input_ids"][0],"".join(uncompleted_answer.split()))
    pos_start = end_of_question - 3 #  XX,XX,?,[,/,INST,]  在这些token上进行研究
    token_range = (pos_start,answer_start_pos)

    if not kind:
        differences = trace_important_states(
            model,
            num_layers,
            batch_input,
            answers_t[1],
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            model,
            num_layers,
            batch_input,
            answers_t[1],
            kind,
            token_range=token_range,
            window=window,
        )

    return dict(
        differences=differences,
        predicted_token=predicted_token,
        counter_factual_token=counter_factual_token,
        answers_t=np.array([elem.item() for elem in answers_t]),
        base_scores=np.array([elem.item() for elem in base_scores]),
        input_ids=batch_input["input_ids"][0].clone().cpu().numpy(),
        input_tokens=decode_tokens(tokenizer, batch_input["input_ids"][0]),
        token_range=token_range,
        window=window,
        kind=kind or "",
    )

def trace_important_window(
    model,
    num_layers,
    inp,
    answer_t,
    kind,
    token_range=None,
    window=10,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    else:
        token_range = range(token_range[0],token_range[1])
    for tnum in tqdm(token_range):
        row = []
        for layer in tqdm(range(num_layers),leave=False):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
            )
            row.append(r)
        table.append(row)
    return np.array(table)



def trace_important_states(
    model,
    num_layers,
    inp,
    answer_t,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    else:
        token_range = range(token_range[0],token_range[1])
    for tnum in tqdm(token_range):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
            )
            row.append(r)
        table.append(row)
    return np.array(table)

def trace_with_patch(
        model,
        inp,
        states_to_patch,
        answer_t,
):
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def patch_rep(x, layer):
        # before the o_proj layer:(batch_size,query_length,num_head * head_dim)
        h = untuple(x)
        for t in patch_spec[layer]:
            h[2,t] = h[1,t] # change the fp component to the tp component
        return x

    with torch.no_grad(), TraceDict(
            model,
            list(patch_spec.keys()),
            edit_output=patch_rep,
    ) as td:
        out = model(
            **inp,
            output_hidden_states=True,
        ) # logits:(batch_size,query_length,vocab_size)
        probs = torch.softmax(out["logits"][:, -1], dim=1)
        ground_truth_token_t = answer_t
        init_ground_truth_score = probs[0,ground_truth_token_t]
        after_ground_truth_score = probs[2,ground_truth_token_t]
        difference = (after_ground_truth_score - init_ground_truth_score).item()
    return difference


def plot_trace_heatmap(trace_result,tokens,fig_size=(8,8),title=None,save_path=None):
    assert len(tokens) == trace_result.shape[0]

    # Increase the size of the figure
    fig = plt.figure(figsize=fig_size)  # Adjust the figure size as needed
    plt.imshow(trace_result, cmap='Blues', aspect='auto')
    # Set labels for x and y axes
    plt.ylabel('Tokens')
    plt.xlabel('Layers')
    curr_layers = [f"layer {i}" for i in range(trace_result.shape[-1])]
    plt.xticks(np.arange(len(curr_layers)), curr_layers,rotation=90)
    # plt.tick_params(axis='x', top=True, labeltop=True,bottom=False,labelbottom=False)
    plt.yticks(np.arange(len(tokens)),tokens)
    # Add a colorbar to show the scale of values
    plt.colorbar()

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices,select
    from utils.nethook import TraceDict,Trace
    import torch.nn as nn
    from tqdm import tqdm
    from experiments.causal_trace.path_patch.single_tok_pred_nobel_prize import format_answer_from_sample
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution,get_distribution_from_hidden_state
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker',action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    use_docker = args.use_docker
    subject_key = 'name'
    answer_key = 'when_fp_question_model_answer'
    question_key = 'when_fp_question'
    fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/hallucinated_samples/NobelPrize/{model_size}'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result figs path {result_figs_dir}...")

    dataset_file = f"{base_dir}/results/causal_trace/tok_pred/NobelPrize/llama2-{model_size}-chat_on_nobel_prize_when_fp_question_token_answer.json"
    samples =read_json(dataset_file)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    # layers = [layername(model,31,'self_attn.o_proj')]

    for i,sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        if sample[token_result_key][0] == True:
            continue
        for kind in ["mlp", "self_attn", None]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/{i}_{sample_name}{kind_suffix}.npz"

            if not os.path.isfile(filename):
                question_reference = sample[question_key]
                question_counter_factual = f'For what specific contribution was {sample["name"]} awarded {sample["categoryFullName"]} in XXXX?'
                uncompleted_answer = format_answer_from_sample(sample)
                # print("Debug Usage")
                false_year = str(sample["awardYear"] + 1)
                true_year = str(sample["awardYear"])
                true_token_ids = tokenizer([true_year], return_tensors='pt')["input_ids"][0][1:]
                false_token_ids = tokenizer([false_year], return_tensors='pt')["input_ids"][0][1:]
                prev_common = []
                for true_token_id, false_token_id in zip(true_token_ids, false_token_ids):
                    if true_token_id != false_token_id:
                        break
                    else:
                        prev_common.append(true_token_id)
                uncompleted_answer += tokenizer.decode(prev_common)
                ground_truth_token_id = true_token_id
                ground_truth_token = tokenizer.decode([ground_truth_token_id])
                # orig_prompt = format_question_answer(question, uncompleted_answer)
                # batch_input = tokenizer(orig_prompt, return_tensors="pt")["input_ids"].to(model.device)

                np_result = calculate_hidden_flow(
                    model,
                    tokenizer,
                    question_reference,
                    question_counter_factual,
                    uncompleted_answer,
                    expect=ground_truth_token,
                    kind=kind,
                )
                if np_result is not None:
                    print(f"Saving to file {filename}")
                    np.savez(filename,**np_result)
                else:
                    print(f"Skip this correct sample {i}{kind_suffix}!")
                    continue
            else:
                np_result = np.load(filename,allow_pickle=True)

            plot_result = dict(np_result)
            trace_result = plot_result["differences"]
            token_range = plot_result["token_range"]
            tokens = plot_result["input_tokens"][token_range[0]:token_range[1]]
            pdf_title = f"{i}_{sample_name}{kind_suffix}"
            pdf_fig_name = f"{i}_{sample_name}{kind_suffix}.pdf"
            pdf_save_file = f'{result_figs_dir}/{pdf_fig_name}'
            plot_trace_heatmap(trace_result,tokens,fig_size=(8,10),title=pdf_title,save_path=pdf_save_file)

        # flag, base_scores, decoded_answers = result
        # sample["base_scores"] = base_scores
        # sample["decoded_answers"] = decoded_answers
        # sample["token_pred"] = flag
        # sample["ground_truth_token"] = ground_truth_token

    # token_predictions = [sample["token_pred"] for sample in samples]
    # token_predictions = [sample["decoded_answers"][0] == sample["ground_truth_token"] for sample in samples]
    # token_acc = sum(token_predictions) / len(samples)
    # print("token_acc:",token_acc)
