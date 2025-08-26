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


# def cal_uncertainty_with_selected_head(
#         selector,
#         question,
#         heads_pos,
#         num_generations
# ):
#     generations, average_neg_log_likelihoods_npy = selector.run_sample_with_corrupted_head(
#         question=question,
#         heads_pos=heads_pos,
#         num_generations=num_generations
#     )
#     semantic_ids_per_sample = get_semantic_ids_per_sample(
#         sample,
#         generation_key,
#         ground_truth_key,
#         judge_per_answer
#     )
#     predictive_entropy_over_concepts = get_predictive_entropy_over_concepts_per_sample(
#         -average_neg_log_likelihoods_npy,
#         semantic_ids_per_sample
#     )
#     return predictive_entropy_over_concepts

# def generate_heads_candidates():

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
    result_dir = f'{base_dir}/results/uncertainty/heads_selection/knockout/NobelPrize/new_{model_size}b'
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
    #     layers = [layername(model, L, 'self_attn.o_proj') for L in range(num_hidden_layers)]
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        filename = f"{path_patch_tok_result}/{i}_{sample_name}.npz"

        if not os.path.exists(filename):
            continue
        total_count += 1
        np_result = dict(np.load(filename, allow_pickle=True))
        score = np_result["differences"]
        if score.max() < score_threshold:
            invalid_count += 1
            continue
        key = f"{i}_{sample_name}"
        base_uncertainty_score = sample_name_to_uncertainty[key]
        top_ks = [elem for elem in top_k_elements(score, k=k) if elem[-1] > score_threshold]
        heads_pos = [(elem[0], elem[1]) for elem in top_ks]

        question = sample[question_key]

        new_file_name = f"{result_dir}/{key}.npz"
        if not os.path.isfile(new_file_name):
            print(f"Preparing for file {new_file_name}...")
            np_result = selector.cal_knock_out_attn_generation(
                sample[ground_truth_key],
                judge_per_answer,
                question,
                num_generations=num_generations,
                base_uncertainty_score=base_uncertainty_score
            )
            print(f"Saving to file {new_file_name}")
            np.savez(new_file_name, **np_result)
        else:
            print(f"Loading from file {new_file_name}")
            np_result = np.load(new_file_name,allow_pickle=True)


        result = dict(np_result)
        token_patch_score = score[:int(score.shape[0]//2),:]
        generation_patch_score = -result["differences"]

        pdf_fig_name = f'{i}_{sample_name}.pdf'
        pdf_save_file = f'{result_figs_dir}/{pdf_fig_name}'

        # plot the two scores
        plot_two_figures(
            token_patch_score,
            generation_patch_score,
            "Tok Pred Based",
            "Generation Based",
            overall_title=key,
            save_path=pdf_save_file
        )

    print("Finished Running!")
    # invalid_rate = invalid_count / total_count
    # print(f"Invalid Scores Rate:{invalid_rate}")
