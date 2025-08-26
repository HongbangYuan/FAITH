import numpy as np
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


def find_token_range(tokenizer, token_array, substring, return_slice=False):
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
        return slice(tok_start, tok_end)
    return (tok_start, tok_end)


def plot_contrasive_scores(scores,colors,labels, title=None, save_path=None,ylimit=None):
    fig, ax = plt.subplots()
    if ylimit:
        plt.ylim(*ylimit)
    # plt.ylim(0, 2.5)
    for i,score in enumerate(scores):
        plot_scores(score, label=labels[i],color='#'+colors[i])
    plt.xlabel('Layers',fontsize=16)
    plt.ylabel('Information Flow',fontsize=16)
    plt.legend(fontsize=16)
    plt.axvline(x=17, color='#fde76b', linestyle='--')
    plt.axvline(x=28, color='#fde76b', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=14)
    if title:
        plt.title(title,fontsize=20)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_scores(scores, label,color):
    data = np.stack(scores)  # (num_samples,num_layers)
    # Use seaborn's lineplot with the mean and confidence interval
    mean_values = np.mean(data, axis=0)
    ci_values = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    # Create x-axis values
    x_values = np.arange(1, data.shape[-1] + 1)
    # Plot the mean line
    plt.plot(x_values, mean_values, label=label,color=color)

    # Fill between the upper and lower bounds of the confidence interval
    plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2,color=color)

    # # Set plot labels and title
    # plt.xlabel('Layers')
    # plt.ylabel('Information Flow')
    # # plt.title('Line Plot with Mean and 95% Confidence Interval')
    #
    # # Show legend
    # plt.legend()

    # Show the plot
    # plt.show()


if __name__ == '__main__':
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.books.books_evaluation import cal_acc_when_fp
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, load_llama_tokenizer
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm

    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    question_key = 'when_false_premise_question2'
    subject_key = 'subject_title'
    score_key = 'entropy_over_concepts'
    ground_truth_key = 'who_question_ground_truth'
    answer_key = 'when_false_premise_question2_model_answer'
    generation_key = 'generations'

    # dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_multiple_generation.json'
    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_new_fp_when_false_premise_question2_model_answer.json'
    result_dir = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/{model_size}'
    samples = read_json(dataset_file)
    acc = cal_acc_when_fp(samples, key=answer_key)
    print("Acc:", acc)
    if debug:
        samples = samples[:10]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    tokenizer = load_llama_tokenizer(model_name_or_path)

    pattern = r"When was the book (?P<book_name>.*?) written by (?P<false_author>.*?)\?"
    results = []
    wrong_sample_scores_book = []
    wrong_sample_scores_author = []
    wrong_sample_scores_question = []
    true_sample_scores_book = []
    true_sample_scores_author = []
    true_sample_scores_question = []
    for idx, sample in enumerate(tqdm(samples[:])):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"
        result = dict(np.load(filename, allow_pickle=True))
        saliency_score = result["saliency_score"]

        question = sample[question_key]
        answer = sample[answer_key]
        orig_prompt = format_question_answer(question, answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"][0]

        match = re.match(pattern, question)
        if match:
            false_author = match.group("false_author")
            book_name = match.group("book_name")
        else:
            raise ValueError(f"Author name and book name not found in sentence {question}")
        # book_range = find_token_range(tokenizer,generation,book_name,return_slice=True)
        # author_range = find_token_range(tokenizer,generation,false_author,return_slice=True)
        question_range = find_token_range(tokenizer, generation, question, return_slice=True)
        _, end_of_question = find_token_range(tokenizer, generation, "[/INST]")
        # info_from_book = saliency_score[:,end_of_question:,book_range].mean(axis=-1).mean(axis=-1)
        # info_from_author = saliency_score[:,end_of_question:,author_range].mean(axis=-1).mean(axis=-1)
        info_from_question = saliency_score[:, end_of_question:, question_range].mean(axis=-1).mean(axis=-1)
        if sample["pred"]:
            # true_sample_scores_book.append(info_from_book)
            # true_sample_scores_author.append(info_from_author)
            true_sample_scores_question.append(info_from_question)
        else:
            # wrong_sample_scores_book.append(info_from_book)
            # wrong_sample_scores_author.append(info_from_author)
            wrong_sample_scores_question.append(info_from_question)

    # book_score_fig_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/{model_name}_focus_on_books.pdf'
    # author_score_fig_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/{model_name}_focus_on_author.pdf'
    question_score_fig_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/{model_name}_focus_on_question.pdf'
    # plot_contrasive_scores(true_sample_scores_book,wrong_sample_scores_book,book_score_fig_file)
    # plot_contrasive_scores(true_sample_scores_author,wrong_sample_scores_author,author_score_fig_file)
    plot_contrasive_scores(true_sample_scores_question, wrong_sample_scores_question, question_score_fig_file)
    # print(f"Saving book score to {book_score_fig_file}")
    # print(f"Saving author score to {author_score_fig_file}")
    print(f"Saving question score to {question_score_fig_file}")

    print("Finished Running!")
