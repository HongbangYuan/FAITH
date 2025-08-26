import os.path

import torch
import numpy as np
import matplotlib.pyplot as plt

llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
llama_prompt_template = llama_sys_prompt + ' {} [/INST] {}'


def remove_subtring(text, signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal, text))
    return text[end_idx:].strip()


def remove_prompt(text):
    return remove_subtring(text, '[/INST]')


def remove_sys_prefix(text):
    return remove_subtring(text, '<</SYS>>')


def format_question_answer(question, answer):
    return llama_prompt_template.format(question, answer)


def truncate(s, tokenizer, max_length=150):
    tokenized_list = tokenizer(s, max_length=max_length, truncation=True).input_ids
    truncated_s = tokenizer.decode(tokenized_list, skip_special_tokens=True)
    return truncated_s


if __name__ == '__main__':
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping
    from core.evaluation.books.books_evaluation import cal_acc_when_fp,cal_acc_who
    from experiments.case_study.find_in_film_release import collect
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    import random

    print("Hello World!")

    model_size = '13b'
    # model_size = '7b'
    debug = False
    question_key = 'when_false_premise_question'
    answer_key = 'when_false_premise_question_model_answer'
    when_fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama-{model_size}-chat_on_books_author_when_false_premise_question_model_answer.json'
    who_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama-{model_size}-chat_on_books_author_who_question_model_answer.json'

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    pp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_who_question_model_answer.json'
    pp_samples = read_json(pp_file)
    acc_who = cal_acc_who(pp_samples)
    print("acc_who",acc_who)

    fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_when_false_premise_question_model_answer.json'
    fp_samples = read_json(fp_file)
    print("Truncating fp answers...")
    for sample in tqdm(fp_samples):
        sample[answer_key] = truncate(sample[answer_key], tokenizer)
    acc_when_fp = cal_acc_when_fp(fp_samples)
    print("acc_when_fp",acc_when_fp)

    results = collect(pp_samples,fp_samples)
    print(" ".join([str(len(result)) for result in results]))

    samples = results[0] + results[3]
    random.shuffle(samples)
    print(f"Select a total of {len(results[0])}+{len(results[3])}={len(samples)} samples")

    pbar = enumerate(tqdm(samples) if not debug else tqdm(samples[:5]))

    average_neg_log_likelihoods = []
    for idx, sample in pbar:
        question = sample[question_key]
        answer = sample[answer_key]
        orig_prompt = format_question_answer(question, answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
        prompt = generation[:, :generation.shape[-1] - len(tokenizer.tokenize(answer))]

        target_ids = generation.clone()
        target_ids[:, :prompt.shape[-1]] = -100
        with torch.no_grad():
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=False)
        average_neg_log_likelihood = model_output['loss'].item()
        average_neg_log_likelihoods.append(average_neg_log_likelihood)

    roc_auc_score = roc_auc_score(
        [not sample["fp_pred"] for sample in samples],
        average_neg_log_likelihoods
    )

    print("roc_auc_score", roc_auc_score)

    print("Finished Running")
    print("Hello!")
