from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch

def jsd_compute(p,q,axis):
    assert p.shape == q.shape
    return distance.jensenshannon(p,q,axis=axis)**2

def get_top_k_from_distribution(tokenizer,distribution,k=10):
    # (num_tokens,num_vocabulary
    return [[tokenizer.decode(t) for t in max_indices(distribution[i],k)] for i in range(distribution.shape[0])]

def get_distribution_from_hidden_state(lm_head,hidden_state):
    # hidden_state(*,token_length,hidden_size)
    return torch.softmax(lm_head(hidden_state),dim=-1).clone().detach().cpu().numpy()
    # return torch.softmax(torch.mm(hidden_state,lm_head.to(hidden_state.device)),dim=-1).clone().detach().cpu().numpy()

def plot_jsds(jsd_npy,layers,fig_size=(8,8),tokens=None,save_path=None):
    if tokens is not None:
        assert len(tokens) == jsd_npy.shape[-1]
    jsd_npy = jsd_npy.transpose()

    # Increase the size of the figure
    fig = plt.figure(figsize=fig_size)  # Adjust the figure size as needed
    plt.imshow(jsd_npy, cmap='Blues', aspect='auto')
    # Set labels for x and y axes
    plt.ylabel('Tokens')
    plt.xlabel('Layers')
    # curr_num_layers = [f"layer {-i-1}" for i in range(jsd_npy.shape[-1])]
    curr_layers = layers
    plt.xticks(np.arange(len(curr_layers)), curr_layers)
    # plt.tick_params(axis='x', top=True, labeltop=True,bottom=False,labelbottom=False)
    plt.yticks(np.arange(len(tokens)),tokens)
    # Add a colorbar to show the scale of values
    plt.colorbar()
    # Rotate the figure by 90 degrees
    # fig.gca().set_transform(plt.gca().transData.rotate_deg(90))


    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json,load_llama_tokenizer,min_indices
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt

    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int,default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=True, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    fp_result_key = 'fp_pred'

    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    result_dir = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/Movies/dola_visualize_{model_size}'
    samples = read_json(dataset_file)
    acc = cal_acc_wiki_movies(samples, key=answer_key,output_key=fp_result_key)
    samples = [sample for sample in samples if sample[fp_result_key] == False]
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    model.train()

    layers = [i for i in range(0,30,2)]
    # layers = [i for i in range(-1,-7,-1)]
    assert max(layers) < model.config.num_hidden_layers

    results = []
    for idx,sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/','').replace(" ",'_')
        filename = f"{result_dir}/{idx}_{sample_name}.pdf"
        answer = sample[answer_key]
        question = sample[question_key]
        orig_prompt = format_question_answer(question, answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
        answer_start_pos = generation.shape[-1] - len(tokenizer.tokenize(remove_prompt(orig_prompt)))
        truncate_length = 50
        answer_range = slice(answer_start_pos-1,answer_start_pos+truncate_length-1)
        plot_answer_range = slice(answer_start_pos,answer_start_pos + truncate_length)

        with torch.no_grad():
            output = model(
                generation.to(model.device),
                output_hidden_states=True,
            )
            distribution_from_logits = torch.softmax(output.logits[0, answer_range, :], dim=-1).cpu().numpy()
            tokens_from_logits = get_top_k_from_distribution(tokenizer,distribution_from_logits,k=25)

            lm_head = model.lm_head
            all_tokens_from_hidden_states = {}
            all_distributions = []
            jsds = []
            for layer in layers:
                distribution_from_hidden_states = get_distribution_from_hidden_state(lm_head, output.hidden_states[layer].squeeze()[answer_range,:])
                tokens_from_hidden_states = get_top_k_from_distribution(tokenizer,distribution_from_hidden_states,k=25)
                all_tokens_from_hidden_states[layer] = tokens_from_hidden_states
                all_distributions.append(distribution_from_hidden_states)
                jsd = jsd_compute(distribution_from_hidden_states,distribution_from_logits,axis=-1)
                jsds.append(jsd)
            jsd_npy = np.stack(jsds)
            if debug:
                filename=None
            plot_jsds(jsd_npy,layers,fig_size=(8,15),tokens=[tokenizer.decode(t) for t in generation[0,plot_answer_range]],save_path=filename)
            # print("Debug Usage")
            # named_modules = {key:value for key,value in list(model.named_modules())[:]}
            # lm_head = model.lm_head
            # selected_hidden_states = torch.stack([output.hidden_states[layer][:,answer_range,:] + 1  for layer in  layers],dim=0).squeeze()
            # last_layer_distribution = torch.softmax(lm_head(output.hidden_states[-1][:,answer_range,:]),dim=-1).cpu().numpy()
            # vocab_distribution = torch.softmax(lm_head(selected_hidden_states),dim=-1).cpu().numpy()

        # jsds = []
        # for i in range(selected_hidden_states.shape[0]):
        #     jsd = jsd_compute(vocab_distribution[i],last_layer_distribution[0],axis=-1)
        #     jsds.append(jsd)
        # print("Debug Usage")


