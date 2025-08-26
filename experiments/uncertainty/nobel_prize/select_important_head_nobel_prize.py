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


def plot_two_figures(score_array1,score_array2,title1,title2,overall_title=None,save_path=None):

    # Create a figure with two subplots
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, axs = plt.subplots(2, 1, figsize=(8, 12)) # vertically
    # fig, axs = plt.subplots(1, 2)

    # Plot the heatmap for the first score array on the left subplot
    axs[0].imshow(score_array1, cmap='Blues', aspect='auto')
    axs[0].set_title(title1)
    axs[0].set_xticks(np.arange(score_array1.shape[1]))
    axs[0].set_yticks(np.arange(score_array1.shape[0]))
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels(np.arange(0, score_array1.shape[0], dtype=int))

    axs[0].set_xlabel('Attention Heads')
    axs[0].set_ylabel('Layers')

    # Plot the heatmap for the second score array on the right subplot
    axs[1].imshow(score_array2, cmap='Blues', aspect='auto')
    axs[1].set_title(title2)
    axs[1].set_xticks(np.arange(score_array2.shape[1]))
    axs[1].set_yticks(np.arange(score_array2.shape[0]))
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels(np.arange(0, score_array2.shape[0], dtype=int))

    axs[1].set_xlabel('Attention Heads')
    axs[1].set_ylabel('Layers')

    # Add colorbars for better interpretation
    cb1 = plt.colorbar(axs[0].imshow(score_array1, cmap='Blues'), ax=axs[0], shrink=0.4)
    cb1.set_label('Tok Score')

    cb2 = plt.colorbar(axs[1].imshow(score_array2, cmap='Blues'), ax=axs[1], shrink=0.4)
    cb2.set_label('Generation Score')

    # Adjust layout for better spacing
    plt.tight_layout()

    if overall_title:
        # Add an overall title to the whole figure
        fig.suptitle(overall_title, fontsize=16)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        # Show the plot
        plt.show()


if __name__ == '__main__':
    from utils import read_pickle
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
    result_dir = f'{base_dir}/results/uncertainty/heads_selection/importance_ranking/NobelPrize/{model_size}b'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result path {result_figs_dir}...")

    question_key = 'when_fp_question'
    generation_key = 'generations'
    ground_truth_key = 'when_answer_ground_truth'
    answer_eval_key = 'when_fp_answer_eval'
    subject_key = 'name'

    samples = load_nobel_prize_only_fp(model_size=args.model_size, use_docker=use_docker)
    sample_name_to_uncertainty = read_pickle(uncertainty_base_score_file)

    head_score_file = f'{result_dir}/heads_score.npy'
    if os.path.isfile(head_score_file):
        head_score = np.load(head_score_file)
        importance_rank = np.argsort(head_score.mean(axis=0))[::-1] + 1
        print("Debug Usage")

    selector = AttnHeadSelector(
        model_size=model_size,
        result_path=result_dir,
        experiment_tag="nobel_prize_uncertainty",
        debug=args.debug,
        use_docker=args.use_docker
    )
    num_generations = 5

    score_threshold = 0.1
    invalid_count = 0
    total_count = 0
    k = 20
    important_heads = []
    head_scores = []
    for i, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        filename = f"{path_patch_tok_result}/{i}_{sample_name}.npz"

        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            continue
        total_count += 1
        np_result = dict(np.load(filename, allow_pickle=True))
        score = np_result["differences"]
        # if score.max() < score_threshold:
        #     invalid_count += 1
        #     continue
        key = f"{i}_{sample_name}"
        base_uncertainty_score = sample_name_to_uncertainty[key]
        top_ks = [elem for elem in top_k_elements(score, k=k) if elem[-1] > score_threshold]

        heads_pos = [(elem[0], elem[1]) for elem in top_ks]
        for elem in top_ks:
            important_heads.append(f"{elem[0]}-{elem[1]}")

        heads_pos = [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)]

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
            curr_scores.append(base_uncertainty_score - curr_uncertainty_score)
        head_scores.append(curr_scores)
        print("Five heads scores:",curr_scores)

    head_scores = np.array(head_scores)
    #
    np.save(head_score_file,head_scores)
    print(f"Saving head scores to {head_score_file}")

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
    #
