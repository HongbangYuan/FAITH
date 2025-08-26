import matplotlib.pyplot as plt

if __name__ == '__main__':
    import torch
    from torch.nn import CrossEntropyLoss
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, select, get_model_name_mapping, load_llama_tokenizer
    from utils.nethook import TraceDict
    from tqdm import tqdm
    # from core.methods.information_flow.saliency_score import format_question_answer,reshape_saliency_score,remove_prompt
    from experiments.information_flow.cal_saliency_score_nobel_prize import reshape_saliency_score ,remove_prompt \
        ,format_question_answer ,untuple
    from experiments.causal_trace.path_patch.single_tok_pred_nobel_prize import format_answer_from_sample
    from experiments.information_flow.information_from_author import find_token_range
    from functools import partial
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=13, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    use_docker = args.use_docker

    subject_key = 'name'
    answer_key = 'when_fp_question_model_answer'
    question_key = 'when_fp_question'
    fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'
    truncate_max_length = 256

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/information_flow/NobelPrize/tok_pred/heads/{model_size}'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result path {result_figs_dir}...")

    samples = load_nobel_prize_only_fp(model_size=args.model_size, use_docker=use_docker)


    results = []
    false_samples_false_object_scores = []
    # false_samples_false_object_scores = []
    for idx, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/' ,'').replace(" " ,'_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"

        results = dict(np.load(filename))
        false_object_scores = results["false_object_score"]
        if not sample[fp_result_key]:
            false_samples_false_object_scores.append(false_object_scores)

    score_npy = np.stack(false_samples_false_object_scores).squeeze().mean(axis=0)
    second_largest = np.partition(score_npy.flatten(), -2)[-2]
    plt.figure(figsize=(8,8))
    plt.title(f"Llama2-chat-{model_size}")
    plt.imshow(score_npy, cmap='Blues', aspect='auto',vmax=second_largest)
    plt.show()

    print("Debug Usage")
