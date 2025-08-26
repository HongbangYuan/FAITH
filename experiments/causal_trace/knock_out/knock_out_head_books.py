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

    book_name = sample["subject_title"]
    author = sample["object_title"]
    answer_template = "According to my knowledge, the  author of the book \"{}\" is "

    uncompleted_answer = answer_template.format(book_name)

    return uncompleted_answer,author


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
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from collections import defaultdict
    from core.methods.causal_trace.casual_trace import find_token_range,untuple
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'subject_title'
    answer_key = 'when_false_premise_question2_model_answer'
    question_key = 'when_false_premise_question2'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'


    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_new_fp_when_false_premise_question2_model_answer.json'
    orig_samples = read_json(dataset_file)
    samples = orig_samples

    acc = cal_acc_when_fp(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers
    # heads_pos = [(1,15),(2,2),(1,22),(5,15),(8,18)]
    heads_pos = [(1, 30), (4, 38), (1, 28), (2, 12), (4, 25)]
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())


    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    pbar = tqdm(samples)
    true_count = 0
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            question = sample[question_key]
            # question = f"Why was the film {sample['movie']} released in XXXX?"
            uncompleted_answer,author = format_answer_from_sample(sample)
            ground_truth_token_id = tokenizer([author],return_tensors='pt')["input_ids"][0][1]
            ground_truth_token = tokenizer.decode([ground_truth_token_id])
            orig_prompt = format_question_answer(question, uncompleted_answer)
            batch_input = tokenizer([orig_prompt], return_tensors="pt", padding=True)
            batch_input = {
                key: value.to(model.device) for key, value in batch_input.items()
            }
            _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0],"".join(question.split()))
            pos = end_of_question - 2  # the previous token position before the question mark.

            def intervene_head(x,layer):
                h = untuple(x)
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
                out = model(
                    **batch_input,
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
    # tok_result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/Books/llama2-{model_size}-chat_on_books_when_fp_question2_token_answer.json"
    # write_to_json(samples,tok_result_file)
    # print(f"Writing to tok result file {tok_result_file}")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
