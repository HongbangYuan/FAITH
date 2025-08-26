import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

def top_k_elements(matrix, k):
    flattened_matrix = matrix.flatten()
    sorted_indices = np.argsort(flattened_matrix)[::-1][:k]
    top_elements = flattened_matrix[sorted_indices]

    # Reshape the indices to get the corresponding row and column indices in the original matrix
    row_indices, col_indices = np.unravel_index(sorted_indices, matrix.shape)

    # Combine row and column indices with corresponding values
    top_elements_with_indices = list(zip(row_indices, col_indices, top_elements))

    return top_elements_with_indices

def sort_elements_by_frequency(elements):
    # Calculate frequencies
    element_frequencies = Counter(elements)

    # Sort elements based on frequencies (first by frequency, then by element)
    sorted_elements = sorted(element_frequencies.items(), key=lambda x: (-x[1], x[0]))

    return sorted_elements

def sort_tuples_by_second_element(tuple_list):
    # Use the sorted function with a lambda function as the key
    sorted_list = sorted(tuple_list, key=lambda x: x[1],reverse=True)
    return sorted_list


if __name__ == '__main__':
    from utils import read_pickle,write_to_pickle
    from core.methods.causal_trace.path_patch_generation import \
        AttnHeadSelector,get_semantic_ids_per_sample,get_predictive_entropy_over_concepts_per_sample
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import os
    import argparse
    from core.evaluation.noble_prize.nobel_prie_when_fp_evaluation import judge_per_answer
    from core.methods.causal_trace.path_patch import plot_path_patch
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    use_docker = args.use_docker
    model_size = args.model_size
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    path_patch_tok_result = f'{base_dir}/results/causal_trace/path_patch/NobelPrize/head_contributions_{model_size}b'
    uncertainty_base_score_file = f'{base_dir}/results/uncertainty/NobelPrize/{model_size}b/llama2-{model_size}b-chat_nobel_prize_uncertainty.pkl'
    result_dir = f'{base_dir}/results/uncertainty/heads_selection/tok_and_generation/NobelPrize/{model_size}b'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result path {result_figs_dir}...")

    score_threshold = 0.01
    all_heads_score_file = f'{result_dir}/all_heads_score_with_threshold_{score_threshold}.pkl'
    # all_heads_score_file = f'{result_dir}/all_heads_score.pkl'
    # print(f"Heads Scores will be stored in file {all_heads_score_file}.")
    if not os.path.isfile(all_heads_score_file):
        print(f"Creating all heads score file {all_heads_score_file}...")

        question_key = 'when_fp_question'
        generation_key = 'generations'
        ground_truth_key = 'when_answer_ground_truth'
        answer_eval_key = 'when_fp_answer_eval'
        subject_key = 'name'

        samples = load_nobel_prize_only_fp(model_size=args.model_size, use_docker=use_docker)
        sample_name_to_uncertainty = read_pickle(uncertainty_base_score_file)

        selector = AttnHeadSelector(
            model_size=model_size,
            result_path=result_dir,
            experiment_tag="nobel_prize_uncertainty",
            debug=args.debug,
            use_docker=args.use_docker
        )
        num_generations = 5

        invalid_count = 0
        total_count = 0
        k = 10
        all_heads_scores = []
        for i, sample in enumerate(tqdm(samples)):
            sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
            filename = f"{path_patch_tok_result}/{i}_{sample_name}.npz"
            if not os.path.exists(filename):
                print(f"File {filename} not found.")
                continue
            total_count += 1
            np_result = dict(np.load(filename, allow_pickle=True))
            score = np_result["differences"]
            if score.max() < score_threshold:
                invalid_count += 1
                continue
            key = f"{i}_{sample_name}"
            base_uncertainty_score = sample_name_to_uncertainty[key]
            top_ks_from_tokens = [elem for elem in top_k_elements(score, k=k) if elem[-1] > 0]
            heads_pos = [(elem[0], elem[1]) for elem in top_ks_from_tokens]

            curr_scores = []
            for head_pos in tqdm(heads_pos,leave=False):
                question_reference = sample[question_key]
                question_counter_factual = f'For what specific contribution was {sample["name"]} awarded {sample["categoryFullName"]} in XXXX?'
                curr_uncertainty_score = selector.cal_path_patch_generation_with_attn_head(
                    sample[ground_truth_key],
                    judge_per_answer,
                    question_reference,
                    question_counter_factual,
                    num_generations=num_generations,
                    head_pos=head_pos
                )
                delta = base_uncertainty_score - curr_uncertainty_score
                curr_scores.append((head_pos,delta))
            sorted_heads = sort_tuples_by_second_element(curr_scores)
            all_heads_scores.append(sorted_heads)
            # print("Debug Usage")
            #
            # for elem in top_ks:
            #     important_heads.append(f"{elem[0]}-{elem[1]}")
            # print("Hello World!")
        write_to_pickle(all_heads_scores,all_heads_score_file)
        print(f"Writing results to {all_heads_score_file}")

    else:
        all_heads_scores = read_pickle(all_heads_score_file)

    # analyze the scores
    print("Debug Usage")

    top = 5
    important_heads = []
    for heads_score in all_heads_scores:
        for elem in heads_score[:top]:
            if elem[-1] > 0:
                important_heads.append(elem[0])

    selected_heads = sort_elements_by_frequency(important_heads)
    print("Heads with corresponding frequencies:")
    for i in range(top):
        print(selected_heads[i])
    print("Selected Heads:")
    print([elem[0] for elem in selected_heads[:top]])

        # heads_pos = [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)]

    #     curr_scores = []
    #     for head_pos in tqdm(heads_pos,leave=False):
    #         question_reference = sample[question_key]
    #         question_counter_factual = f'For what specific contribution was {sample["name"]} awarded {sample["categoryFullName"]} in XXXX?'
    #         curr_uncertainty_score = selector.cal_path_patch_generation_with_attn_head(
    #             sample[ground_truth_key],
    #             judge_per_answer,
    #             question_reference,
    #             question_counter_factual,
    #             num_generations=num_generations,
    #             head_pos=head_pos
    #         )
    #         curr_scores.append(base_uncertainty_score - curr_uncertainty_score)
    #     head_scores.append(curr_scores)
    #     print("Five heads scores:",curr_scores)
    #
    # head_scores = np.array(head_scores)
    #
    # print(f"Analyze a total of {len(samples)} samples.")
    # invalid_rate = invalid_count / len(samples)
    # print(f"Invalid Scores Rate:{invalid_rate}")
    # heads = sort_elements_by_frequency(important_heads)
    #
    # def f(x):
    #     return int(x.split('-')[0]), int(x.split('-')[1])
    #
    # selected_heads = []
    # print(f"Top heads and frequencies under threshold {score_threshold}:")
    # for i in range(k):
    #     print(heads[i])
    #     selected_heads.append(f(heads[i][0]))

