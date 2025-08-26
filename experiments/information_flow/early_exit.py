from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch


def format_answer_from_sample(sample):
    # answer_template = "I apologize, but the film \"{}\" was not released in {}. The film was actually released in {}"
    # answer_template = "I apologize, but the film \"{}\" was actually released in {}"
    answer_template = "According to my knowledge, the film \"{}\" was released in {}"
    film_name = sample["movie"]
    false_year = sample["false_year"]
    true_year = str(sample["time"][0])
    uncompleted_answer = answer_template.format(film_name, true_year[:-1])
    ground_truth_token = true_year[-1]
    # return answer_template.format(film_name, false_year, true_year[:-1]),true_year[-1]
    return uncompleted_answer, ground_truth_token

def plot_contrasive_scores(true_scores,false_scores,layers,fig_size=(8,8),save_path=None,title=None):
    plt.figure(figsize=fig_size)
    plot_scores(true_scores,layers,label='Ground Truth Token')
    plot_scores(false_scores,layers,label="Predicted Token")
    if title:
        plt.title(title)
    plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_scores(scores,layers,label):
    data = -scores[:,::-1]  # (num_samples,num_layers)
    # Use seaborn's lineplot with the mean and confidence interval
    mean_values = np.mean(data, axis=0)
    ci_values = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    # Create x-axis values
    # x_values = np.arange(1, data.shape[-1] + 1)
    x_values = np.arange(0, data.shape[-1])
    # Plot the mean line
    plt.plot(x_values, mean_values, label=label)

    # Fill between the upper and lower bounds of the confidence interval
    plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2)

    # Set plot labels and title
    plt.xlabel('Layers')
    plt.ylabel('Ranks')

    curr_layers = layers[::-1]
    plt.xticks(np.arange(len(curr_layers)), curr_layers)

    # # Get the current y-axis tick labels
    # current_y_labels =  plt.yticks()[0][:-1]
    #
    # # Convert the labels to numerical values, negate them, and convert back to strings
    # new_y_labels = [str(-int(label)) for label in current_y_labels][:-1] + ['1']
    # current_y_labels = [int(label) for label in current_y_labels]
    # # Set the new y-axis tick labels
    # plt.yticks(current_y_labels, new_y_labels)

    # plt.title('Line Plot with Mean and 95% Confidence Interval')

    # Show legend
    plt.legend()


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

    # # layers = [-i for i in range(1,32 + 1 - 5)]
    layers = [-i for i in range(1,7)]

    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    for idx, sample in enumerate(tqdm(samples)):
        with torch.no_grad():
            question = sample[question_key]
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
                if predicted_token == ground_truth_token:
                    predicted_token = str(sample["false_year"])[-1]
                    predicted_token_score_layers = []
                    predicted_token_rank_layers = []
                    ground_truth_token_score_layers = []
                    ground_truth_token_rank_layers = []
                    for layer in layers:
                        distribution = get_distribution_from_hidden_state(model.lm_head,out["hidden_states"][layer][:,-1,:])
                        # distribution = probs.clone().cpu().numpy()
                        top_indices = np.argsort(distribution, axis=-1)[0][::-1]
                        ground_truth_token_rank = np.where(top_indices == tokenizer.convert_tokens_to_ids([ground_truth_token]))[0][0] + 1
                        ground_truth_token_score = distribution[0, ground_truth_token_rank - 1]
                        predicted_token_rank = np.where(top_indices == tokenizer.convert_tokens_to_ids([predicted_token]))[0][0] + 1
                        predicted_token_score = distribution[0, predicted_token_rank - 1]
                        ground_truth_token_score_layers.append(ground_truth_token_score)
                        ground_truth_token_rank_layers.append(ground_truth_token_rank)
                        predicted_token_rank_layers.append(predicted_token_rank)
                        predicted_token_score_layers.append(predicted_token_score)

                        # rank_in_false_samples.append(ground_truth_token_rank)
                        # score_in_false_samples.append(ground_truth_token_score)
                        # k = 20
                        # top_indices = top_indices[:k]
                        # top_scores = distribution[:,top_indices]
                        # top_tokens = [tokenizer.decode([t]) for t in top_indices]
                    ground_truth_ranks.append(ground_truth_token_rank_layers)
                    predicted_token_ranks.append(predicted_token_rank_layers)
                    # print('\n')
                    # print(predicted_token_rank_layers)
                    # print(ground_truth_token_rank_layers)
                    # print("Debug Usage!")
            sample[token_result_key] = (ground_truth_token == predicted_token,predicted_token,base_score)
            results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])
    ground_truth_ranks = np.array(ground_truth_ranks)
    predicted_token_ranks = np.array(predicted_token_ranks)

    np.save("ground_truth_ranks.npy",ground_truth_ranks)
    np.save("predicted_token_ranks.npy",predicted_token_ranks)
    # ground_truth_ranks = np.load("ground_truth_ranks.npy")
    # predicted_token_ranks = np.load("predicted_token_ranks.npy")

    # plot the token ranks ranging with layers
    print("Debug Usage")
    # plot_contrasive_scores(np.array(ground_truth_ranks),np.array(predicted_token_ranks))
    plot_contrasive_scores(ground_truth_ranks,predicted_token_ranks,layers)


    # token_predictions = [sample[token_result_key][0] for sample in samples]
    # token_acc = sum(token_predictions) / len(results)
    # print("token_acc:",token_acc)
    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample for sample in samples if sample[token_result_key][0] == True and sample[fp_result_key] == False]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
