from core.methods.causal_trace.casual_trace import find_token_range
from core.methods.causal_trace.causal_trace_tok_pred import predict_from_input,decode_tokens
import matplotlib.pyplot as plt
from core.methods.information_flow.saliency_score import format_question_answer
import torch
from utils.nethook import TraceDict
from tqdm import tqdm
from core.methods.causal_trace.causal_trace_tok_pred import layername
import numpy as np
import os

def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def calculate_path_patch(
        model,
        tokenizer,
        question_reference,
        question_counter_factual,
        uncompleted_answer,
        ground_truth_token=None,
):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers

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
    pos = end_of_question - 2 # the previous token position before the question mark.
    layers = [layername(model,L,'self_attn.o_proj') for L in range(num_hidden_layers)]

    differences = []
    for selected_layer in tqdm(layers):
        row = []
        for selected_head in tqdm(range(num_heads),leave=False):
            dim_start = selected_head * head_dim
            dim_end = (selected_head + 1) * head_dim

            def patch_rep(x, layer):
                # before the o_proj layer:(batch_size,query_length,num_head * head_dim)
                h = untuple(x)
                # keeping all the heads frozen to there activations on reference data
                h[2,pos,:] = h[0,pos,:]
                if layer in selected_layer:
                    # except for the sender head whose activation is on counter factual data
                    h[2, pos, dim_start:dim_end] = h[1, pos, dim_start:dim_end]
                return x

            with torch.no_grad(), TraceDict(
                    model,
                    layers,
                    edit_input=patch_rep,
            ) as td:
                out = model(
                    **batch_input,
                    output_hidden_states=True,
                ) # logits:(batch_size,query_length,vocab_size)
                probs = torch.softmax(out["logits"][:, -1], dim=1)
                ground_truth_token_t = answers_t[1]
                init_ground_truth_score = probs[0,ground_truth_token_t]
                after_ground_truth_score = probs[2,ground_truth_token_t]
                difference = (after_ground_truth_score - init_ground_truth_score).item()
            row.append(difference)
        differences.append(row)
    differences = np.array(differences)
    return dict(
        differences = differences,
        predicted_token = predicted_token,
        counter_factual_token = counter_factual_token,
        answers_t = np.array([elem.item() for elem in answers_t]),
        base_scores = np.array([elem.item() for elem in base_scores]),
    )
    # base_scores, answers = torch.max(probs, dim=1)
    # decoded_answers = [tokenizer.decode([t]) for t in answers]
    # print(decoded_answers)
    # print(base_scores)
    # print("Ground Truth Token:",ground_truth_token)
    # print("Hello World!")


def plot_path_patch(scores,fig_size=(8,8),title=None,save_path=None):
    # scores shape: (num_layers,num_heads)
    fig = plt.figure(figsize=fig_size)  # Adjust the figure size as needed
    plt.imshow(scores, cmap='Blues', aspect='auto')

    plt.colorbar()

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, select
    from experiments.causal_trace.single_tok_pred import format_answer_from_sample
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker',action='store_true',)
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'
    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/path_patch/Movies/head_contributions_13b'
    result_figs_dir = f'{result_dir}/figs'
    assert os.path.exists(result_figs_dir)
    assert os.path.exists(result_dir)

    # dataset_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer_false.json"
    # dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer.json'
    orig_samples = read_json(dataset_file)
    samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    samples = [sample for sample in samples_one_time if
                        str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    # layers = [layername(model,31,'self_attn.o_proj')]

    for i,sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        filename = f"{result_dir}/{i}_{sample_name}.npz"

        if sample["token_pred"][0]:
            continue

        if not os.path.isfile(filename):
            print(f"Processing sample {i} {sample_name}")
            question_reference = sample[question_key]
            question_counter_factual = f"Why was the film {sample['movie']} released in XXXX?"
            uncompleted_answer, ground_truth_token = format_answer_from_sample(sample)

            np_result = calculate_path_patch(
                model,
                tokenizer,
                question_reference,
                question_counter_factual,
                uncompleted_answer,
                ground_truth_token=ground_truth_token,
            )
            if np_result is not None:
                print(f"Saving to file {filename}")
                np.savez(filename,**np_result)
            else:
                print(f"Skip this correct sample {i}!")
                continue
        else:
            np_result = np.load(filename,allow_pickle=True)


        pdf_fig_name = f'{i}_{sample_name}.pdf'
        pdf_save_file = f'{result_figs_dir}/{pdf_fig_name}'
        pdf_title = f"{i}_{sample_name}"
        result = dict(np_result)
        plot_path_patch(result["differences"],title=pdf_title,save_path=pdf_save_file)
        # print("Debug Usage")

        # flag, base_scores, decoded_answers = result
        # sample["base_scores"] = base_scores
        # sample["decoded_answers"] = decoded_answers
        # sample["token_pred"] = flag
        # sample["ground_truth_token"] = ground_truth_token

    # token_predictions = [sample["token_pred"] for sample in samples]
    # token_predictions = [sample["decoded_answers"][0] == sample["ground_truth_token"] for sample in samples]
    # token_acc = sum(token_predictions) / len(samples)
    # print("token_acc:",token_acc)
