import os.path

import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.evaluation.books.books_evaluation import check_who

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



def get_average_neg_log_likelihood(model,tokenizer,samples):
    pbar = enumerate(tqdm(samples) if not debug else tqdm(samples[:5]))

    average_neg_log_likelihoods = []
    for idx, sample in pbar:
        question = sample[question_key]
        average_neg_log_likelihoods_per_sample = []
        for answer in sample[generation_key]:
            answer = answer[0]
            orig_prompt = format_question_answer(question, answer)
            generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
            prompt = generation[:, :generation.shape[-1] - len(tokenizer.tokenize(answer))]

            target_ids = generation.clone()
            target_ids[:, :prompt.shape[-1]] = -100
            with torch.no_grad():
                model_output = model(generation.to(model.device), labels=target_ids.to(model.device),
                                     output_hidden_states=False)
            average_neg_log_likelihood = model_output['loss'].item()
            average_neg_log_likelihoods_per_sample.append(average_neg_log_likelihood)
        average_neg_log_likelihoods.append(average_neg_log_likelihoods_per_sample)
    average_neg_log_likelihoods_npy = np.array(average_neg_log_likelihoods)
    return average_neg_log_likelihoods_npy

def get_semantic_ids(samples,generation_key,check_who=check_who):
    semantic_ids = []
    for sample in samples:
        generations = sample[generation_key]
        flags = [check_who(sample["who_question_ground_truth"],g[0]) for g in generations]
        semantic_ids_per_sample = []
        count = 2
        for flag in flags:
            if flag:
                semantic_ids_per_sample.append(1)
            else:
                semantic_ids_per_sample.append(count)
                count = count+1
        semantic_ids.append(semantic_ids_per_sample)
    return semantic_ids

def get_predictive_entropy_over_concepts(log_likelihods,semantic_set_ids):
    print("debug Usage")
    llh_shift = torch.tensor(0.0)
    entropies = []
    for row_index in range(log_likelihods.shape[0]):
        row = torch.tensor(log_likelihods[row_index])
        aggregated_likelihoods = []
        semantic_set_ids_row = torch.tensor(semantic_set_ids[row_index])
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy)
    return np.array(entropies)

if __name__ == '__main__':
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json
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
    generation_key = 'generations'

    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_multiple_generation.json'
    numpy_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_multiple_generation_average_neg_log_likelihoods.npy"
    entropy_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_base_entropy.json"

    samples = read_json(result_file)
    semantic_ids = get_semantic_ids(samples,generation_key)

    if not os.path.isfile(numpy_file):
        model_name = "llama2-{}-chat".format(model_size)
        model_name_or_path = model_name_mapping[model_name]
        print("Model name:", model_name)
        model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

        average_neg_log_likelihoods_npy = get_average_neg_log_likelihood(model,tokenizer,samples)
        np.save(numpy_file, average_neg_log_likelihoods_npy)
    else:
        print(f"Loading pre-calculated numpy array from {numpy_file}")
        average_neg_log_likelihoods_npy = np.load(numpy_file)

    predictive_entropy = -np.sum(-average_neg_log_likelihoods_npy, axis=1) / average_neg_log_likelihoods_npy.shape[1]
    # mean_across_models = torch.logsumexp(torch.tensor(-average_neg_log_likelihoods_npy).view(1,-1,5),dim=0) - torch.log(torch.tensor(1))

    predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-average_neg_log_likelihoods_npy,semantic_ids)

    score = roc_auc_score(
        [not sample["fp_pred"] for sample in samples[:average_neg_log_likelihoods_npy.shape[0]]],
        average_neg_log_likelihoods_npy.mean(axis=-1)
    )

    predictive_entropy_over_concepts_score = roc_auc_score(
        [not sample["fp_pred"] for sample in samples[:average_neg_log_likelihoods_npy.shape[0]]],
        predictive_entropy_over_concepts
    )

    print("roc_auc_score", score)
    print("predictive_entropy_over_concepts_score",predictive_entropy_over_concepts_score)

    for sample,entropy,entropy_over_concepts in zip(samples,predictive_entropy.tolist(),predictive_entropy_over_concepts.tolist()):
        sample["entropy"] = entropy
        sample["entropy_over_concepts"] = entropy_over_concepts
    write_to_json(samples,entropy_file)

    print("Finished Running")
    print("Hello!")
