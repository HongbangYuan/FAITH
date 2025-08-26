import numpy as np
import torch

from core.methods.information_flow.saliency_score import format_question_answer
import re
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring,return_slice=False):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index("".join(substring.split()))
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len("".join(substring.split())):
            tok_end = i + 1
            break
    assert "".join(toks[tok_start:tok_end]) == "".join(substring.split())
    if return_slice:
        return slice(tok_start,tok_end)
    return (tok_start, tok_end)

def plot_contrasive_scores(true_scores,false_scores,save_path=None):
    plt.figure()
    plot_scores(true_scores,label='TrueSamples')
    plot_scores(false_scores,label="FalseSamples")
    plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_scores(scores,label):
    data = np.stack(scores) # (num_samples,num_layers)
    # Use seaborn's lineplot with the mean and confidence interval
    mean_values = np.mean(data, axis=0)
    ci_values = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    # Create x-axis values
    x_values = np.arange(1, data.shape[-1] + 1)
    # Plot the mean line
    plt.plot(x_values, mean_values, label=label)

    # Fill between the upper and lower bounds of the confidence interval
    plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2, label='95% CI')

    # Set plot labels and title
    plt.xlabel('Layers')
    plt.ylabel('Information Flow')
    # plt.title('Line Plot with Mean and 95% Confidence Interval')

    # Show legend
    plt.legend()

    # Show the plot
    # plt.show()


if __name__ == '__main__':
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.books.books_evaluation import cal_acc_when_fp
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json,load_llama_tokenizer,select
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm
    from einops import rearrange

    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--alpha',type=float,default=2)
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    question_key = 'when_false_premise_question2'
    subject_key = 'subject_title'
    score_key = 'entropy_over_concepts'
    ground_truth_key = 'who_question_ground_truth'
    answer_key = 'when_false_premise_question2_model_answer'
    generation_key = 'generations'
    new_answer_key = 'when_false_premise_question2_model_answer_after_intervene'
    pattern = r"When was the book (?P<book_name>.*?) written by (?P<false_author>.*?)\?"
    alpha = args.alpha
    print(f"Intervention strength:{alpha}")

    # dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_multiple_generation.json'
    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_new_fp_when_false_premise_question2_model_answer.json'
    samples = read_json(dataset_file)
    if debug:
        samples = samples[:50]
    before_acc = cal_acc_when_fp(samples, key=answer_key,output_key='before_pred')
    print("Before Intervene Acc:", before_acc)


    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)
    layers_to_intervene = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(10,20+1)]
    def intervene(x,layer_name,e_range):
        h = untuple(x) # (batch_size,sequence_length,num_head * head_dim)
        if h.shape[1] == 1:
            # Pay close attention when prompt length=1! But I don't think this would happen.
            return x
        h[:,e_range,:] = h[:,e_range,:] * alpha
        return x

    model_answers = []
    for idx, sample in enumerate(tqdm(samples[:])):
        question = sample[question_key]
        answer = sample[answer_key]
        orig_prompt = format_question(sample[question_key], instruction=""),

        match = re.match(pattern, question)
        if match:
            false_author = match.group("false_author")
            book_name = match.group("book_name")
        else:
            raise ValueError(f"Author name and book name not found in sentence {question}")

        batch_input = tokenizer(orig_prompt, return_tensors='pt')
        book_range = find_token_range(tokenizer, batch_input["input_ids"][0], book_name, return_slice=True)

        with torch.no_grad(), TraceDict(
                model,
                layers_to_intervene,
                edit_output=partial(intervene,e_range=book_range)
        ) as ret:
            output = model.generate(
                input_ids=batch_input["input_ids"].to(model.device),
                attention_mask=batch_input["attention_mask"].to(model.device),
                max_new_tokens=256,  # Define the maximum length for decoding
                num_beams=5,
                do_sample=False
            )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        new_answer = remove_prompt(decoded_output[0])
        sample[new_answer_key] = new_answer
    print(f"Before Intervene:before_acc={before_acc}")
    after_acc = cal_acc_when_fp(samples,new_answer_key,output_key='after_pred')
    print(f"After Intervene:after_acc={after_acc}")

    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/intervention/llama2-{model_size}-chat_alpha={alpha}_on_books_{before_acc}_to_{after_acc}.json'
    write_to_json(samples,result_file)
    print(f"Writing results to {result_file}")

    analyze_samples = [sample for sample in samples if sample["before_pred"] != sample["after_pred"]]

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../ python subject_strengthen.py --model_size 7 --alpha -2 --debug
