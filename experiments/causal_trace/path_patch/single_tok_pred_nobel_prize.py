from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch
import re

def format_answer_from_sample(sample):
    # question = sample["when_false_premise_question"]
    # pattern = r"When did (?P<false_author>.*?) write the book (?P<book_name>.*?)\?"
    # match = re.match(pattern, question)
    # if match:
    #     false_author = match.group("false_author")
    #     book_name = match.group("book_name")
    # else:
    #     raise ValueError(f"Book name and author not found in {question}!")


    answer_template = "According to my knowledge, {} won the {} in "
    uncompleted_answer = answer_template.format(sample["name"],sample["categoryFullName"])

    return uncompleted_answer


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
    from core.evaluation.books.books_evaluation import cal_acc_when_fp
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'name'
    answer_key = 'when_fp_question_model_answer'
    question_key = 'when_fp_question'
    fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'

    samples = load_nobel_prize_only_fp(model_size=args.model_size)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    pbar = tqdm(samples)
    true_count = 0
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            # question = sample[question_key]
            question = f'For what specific contribution was {sample["name"]} awarded {sample["categoryFullName"]} in XXXX?'
            uncompleted_answer = format_answer_from_sample(sample)
            # print("Debug Usage")
            false_year = str(sample["awardYear"] + 1)
            true_year = str(sample["awardYear"])
            true_token_ids = tokenizer([true_year],return_tensors='pt')["input_ids"][0][1:]
            false_token_ids = tokenizer([false_year],return_tensors='pt')["input_ids"][0][1:]
            prev_common = []
            for true_token_id,false_token_id in zip(true_token_ids,false_token_ids):
                if true_token_id != false_token_id:
                    break
                else:
                    prev_common.append(true_token_id)
            uncompleted_answer += tokenizer.decode(prev_common)
            ground_truth_token_id = true_token_id
            ground_truth_token = tokenizer.decode([ground_truth_token_id])
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
            true_count += (ground_truth_token == predicted_token)
            pbar.set_description(f"{true_count}/{idx+1} acc={true_count/(idx+1):.4f}")
            results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])

    token_predictions = [sample[token_result_key][0] for sample in samples]
    token_acc = sum(token_predictions) / len(results)
    print("token_acc:",token_acc)
    # analyze_samples = [sample if sample[token_result_key][0] == False else None for sample in samples]
    # analyze_samples = [sample for sample in samples  if sample[token_result_key][0] == False]
    # tok_result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/NobelPrize/llama2-{model_size}-chat_on_nobel_prize_when_fp_question_token_answer.json"
    # write_to_json(samples,tok_result_file)
    # print(f"Writing to tok result file {tok_result_file}")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
