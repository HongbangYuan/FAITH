from scipy.spatial import distance
from utils import max_indices, get_model_name_mapping
import matplotlib.pyplot as plt
import torch
import re

def format_answer_from_sample(sample,model_size):
    # question = sample["when_false_premise_question"]
    # pattern = r"When did (?P<false_author>.*?) write the book (?P<book_name>.*?)\?"
    # match = re.match(pattern, question)
    # if match:
    #     false_author = match.group("false_author")
    #     book_name = match.group("book_name")
    # else:
    #     raise ValueError(f"Book name and author not found in {question}!")


    # if model_size == 13:
    answer_template = "Hello! I'm here to help you answer your question. The film \"{}\" was released in "
    # else:
    #     answer_template = "According to my knowledge, {} was released in "

    uncompleted_answer = answer_template.format(sample["movie"])

    return uncompleted_answer


if __name__ == '__main__':
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
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution,get_distribution_from_hidden_state
    from core.evaluation.books.books_evaluation import cal_acc_when_fp
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    from collections import defaultdict
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()


    model_size = f'{args.model_size}b'
    use_docker = args.use_docker
    debug = args.debug
    subject_key = 'movie'
    # answer_key = 'when_fp_question_model_answer'
    question_key = 'fp_question_'
    # fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'
    ground_truth_key = 'ground_truth_token'

    print("question key:",question_key)
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    dataset_file = f'{base_dir}/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_more_fp_fp_question_4_model_answer.json'
    result_dir = f'{base_dir}/results/causal_trace/tok_pred'
    if not os.path.exists(result_dir):
        print(f"Creating the result path {result_dir}")
        os.makedirs(result_dir)

    # samples = load_nobel_prize_only_fp(model_size=args.model_size)
    orig_samples = read_json(dataset_file)
    samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    samples = [sample for sample in samples_one_time if
                        str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]
    print(f"Selected {len(samples)} Samples:")

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
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
    true_count = defaultdict(int)
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            uncompleted_answer = format_answer_from_sample(sample,args.model_size)
            # print("Debug Usage")
            false_year = str(min(sample["time"]) - 1)
            true_year = str(min(sample["time"]))
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

            template1 = f"Why was the film {sample['movie']} released in XXXX?"
            template2 = f"What was the film {sample['movie']} released in XXXX describing about?"
            template3 = f"What was the XXXX film {sample['movie']} about?"
            template4 = f"Who are the main characters in the XXXX film {sample['movie']}?"
            question_number_to_counter_factual = {
                1: template1,
                2: template2,
                3: template3,
                4: template4,
            }

            selected = [1,2,3,4]
            for k in selected:
                curr_question_key = question_key + str(k)
                curr_tok_result_key = token_result_key + str(k)
                curr_ground_truth_key = ground_truth_key + str(k)

                # question = question_number_to_counter_factual[k]
                question = sample[curr_question_key]

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
                sample[curr_ground_truth_key] = ground_truth_token
                sample[curr_tok_result_key] = (ground_truth_token == predicted_token,predicted_token,base_score.item())
                true_count[k] += (ground_truth_token == predicted_token)
                # true_count += (ground_truth_token == predicted_token)
            pbar.set_description("  ".join(f"acc={true_count[k]/(idx+1):.6f}" for k in selected))
            # results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])

    # token_predictions = [sample[token_result_key][0] for sample in samples]
    # token_acc = sum(token_predictions) / len(results)
    # print("token_acc:",token_acc)
    # analyze_samples = [sample if sample[token_result_key][0] == False else None for sample in samples]
    # analyze_samples = [sample for sample in samples  if sample[token_result_key][0] == False]
    tok_result_file = f"{result_dir}/llama2-{model_size}-chat_on_wiki_movie_more_fps_token_answer.json"
    write_to_json(samples,tok_result_file)
    print(f"Writing to tok result file {tok_result_file}")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
