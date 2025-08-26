import os
from core.methods.causal_trace.path_patch import plot_path_patch


def is_file(filepath):
    return os.path.isfile(filepath)


def get_files_except_dicts(directory):
    files = [f for f in os.listdir(directory) if is_file(os.path.join(directory, f))]
    return files


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


if __name__ == '__main__':
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices, select
    from utils.nethook import TraceDict, Trace
    import torch.nn as nn
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from experiments.information_flow.dola_visualize import get_top_k_from_distribution, \
        get_distribution_from_hidden_state
    from collections import Counter
    import argparse
    import os

    # result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/path_patch/NobelPrize/head_contributions_7b'
    # result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/path_patch/NobelPrize/more_fps/head_contributions_13b/question_4'
    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/path_patch/NobelPrize/more_fps/head_contributions_b/question_4'
    files = get_files_except_dicts(result_dir)
    print(f"Analyze a total of {len(files)} samples.")
    count = 0
    score_threshold = 0.1

    k = 5
    important_heads = []
    for file in files:
        file_path = os.path.join(result_dir, file)
        np_result = dict(np.load(file_path, allow_pickle=True))
        # difference score: (num_layers,num_heads)
        score = np_result["differences"]
        if score.max() < score_threshold:
            count += 1
            continue
        top_ks = top_k_elements(score, k=k)
        top_ks = [elem for elem in top_ks if elem[-1] > score_threshold]
        # plot_path_patch(score)
        for elem in top_ks:
            important_heads.append(f"{elem[0]}-{elem[1]}")


    invalid_rate = count / len(files)
    print(f"Invalid Scores Rate:{invalid_rate}")

    heads = sort_elements_by_frequency(important_heads)

    def f(x):
        return int(x.split('-')[0]), int(x.split('-')[1])

    selected_heads = []
    print(f"Top heads and frequencies under threshold {score_threshold}:")
    for i in range(k):
        print(heads[i])
        selected_heads.append(f(heads[i][0]))

    print(f"Selected top {k} important heads:")
    print(selected_heads)