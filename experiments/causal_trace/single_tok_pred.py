from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch


def format_answer_from_sample(sample):
    # answer_template = "I apologize, but the film \"{}\" was not released in {}. The film was actually released in {}"
    # answer_template = "I apologize, but the film \"{}\" was actually released in {}"
    # answer_template = "According to my knowledge, the film \"{}\" was released in {}"
    answer_template = "Hello! I'm here to help you answer your question. The film \"{}\" was released in {}"
    film_name = sample["movie"]
    false_year = sample["false_year"]
    true_year = str(sample["time"][0])
    uncompleted_answer = answer_template.format(film_name,true_year[:-1])
    # uncompleted_answer = answer_template.format(true_year[:-1])
    ground_truth_token = true_year[-1]
    # return answer_template.format(film_name, false_year, true_year[:-1]),true_year[-1]
    return uncompleted_answer, ground_truth_token


if __name__ == '__main__':
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
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution,get_distribution_from_hidden_state
    import argparse

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

    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    orig_samples = read_json(dataset_file)
    samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    samples = [sample for sample in samples_one_time if
                        str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]

    acc = cal_acc_wiki_movies(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    layers = [-i for i in range(1,32 + 1 - 5)]

    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    for idx, sample in enumerate(tqdm(samples)):
        with torch.no_grad():
            question = sample[question_key]
            # question = f"Why was the film {sample['movie']} released in XXXX?"
            uncompleted_answer,ground_truth_token = format_answer_from_sample(sample)
            orig_prompt = format_question_answer(question, uncompleted_answer)
            batch_input = tokenizer(orig_prompt, return_tensors="pt")["input_ids"].to(model.device)

            with torch.no_grad():
                out = model(
                    batch_input,
                    output_hidden_states=True,
                )
                probs = torch.softmax(out["logits"][:, -1], dim=1)
                base_score, answer = torch.max(probs, dim=1)
                predicted_token = tokenizer.decode(answer)
            sample["ground_truth_token"] = ground_truth_token
            sample[token_result_key] = (ground_truth_token == predicted_token,predicted_token,base_score.item())
            results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])

    token_predictions = [sample[token_result_key][0] for sample in samples]
    token_acc = sum(token_predictions) / len(results)
    print("token_acc:",token_acc)
    # analyze_samples = [sample if sample[token_result_key][0] == False else None for sample in samples]
    # analyze_samples = [sample for sample in samples  if sample[token_result_key][0] == False]
    # tok_result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer.json"
    # write_to_json(samples,tok_result_file)
    # print(f"Writing to tok result file {tok_result_file}")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
