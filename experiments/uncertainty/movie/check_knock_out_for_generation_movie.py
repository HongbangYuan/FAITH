import numpy as np
import torch
import matplotlib.pyplot as plt

def top_k_elements(matrix, k):
    flattened_matrix = matrix.flatten()
    sorted_indices = np.argsort(flattened_matrix)[::-1][:k]
    top_elements = flattened_matrix[sorted_indices]

    # Reshape the indices to get the corresponding row and column indices in the original matrix
    row_indices, col_indices = np.unravel_index(sorted_indices, matrix.shape)

    # Combine row and column indices with corresponding values
    top_elements_with_indices = list(zip(row_indices, col_indices, top_elements))

    return top_elements_with_indices


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
    from utils import read_pickle,read_json
    from core.methods.causal_trace.path_patch_generation import \
        AttnHeadSelector,get_semantic_ids_per_sample,get_predictive_entropy_over_concepts_per_sample
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import os
    import argparse
    from core.evaluation.noble_prize.nobel_prie_when_fp_evaluation import judge_per_answer
    from core.methods.causal_trace.path_patch import plot_path_patch

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    question_key = 'why_fp_question'
    generation_key = 'generations'
    ground_truth_key = 'time'
    answer_key = 'why_fp_question_model_answer'
    answer_eval_key = 'pred'
    subject_key = 'movie'
    num_generations = 10

    model_size = args.model_size
    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    dataset_dir = f'{base_dir}/results/uncertainty/Movie'
    dataset_file = f'{dataset_dir}/llama2-{model_size}b-chat_on_movie_multiple_generation.json'
    result_dir = f'{base_dir}/results/uncertainty/Movie/{model_size}b'
    result_file = f'{result_dir}/llama2-{model_size}b-chat_movie_uncertainty.pkl'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")
    samples = read_json(dataset_file)


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
    k = 10
    for i, sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')

        question_reference = sample[question_key]
        results = selector.cal_knock_out_attn_generation_with_single_head(
            sample[ground_truth_key],
            judge_per_answer,
            question_reference,
            num_generations=num_generations,
            head_pos=(1,22),
            return_generations=True
        )
