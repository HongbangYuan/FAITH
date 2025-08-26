import torch.autograd
import numpy as np


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
llama_prompt_template = llama_sys_prompt + ' {} [/INST] {}'


def remove_subtring(text, signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal, text))
    # return text[end_idx:].strip()
    return text[end_idx:]


def remove_prompt(text):
    return remove_subtring(text, '[/INST]')


def remove_sys_prefix(text):
    return remove_subtring(text, '<</SYS>>')


def format_question_answer(question, answer):
    return llama_prompt_template.format(question, answer)


def reshape_saliency_score(saliency_score, orig_prompt, tokenizer):
    # saliency_score:(num_layers,num_orig_prompt_tokens,num_orig_prompt_tokens) (32,221,221)

    prompt = remove_sys_prefix(orig_prompt)
    prefix_token_list = tokenizer.tokenize(llama_sys_prompt)
    new_saliency_score = saliency_score[:, len(prefix_token_list):, len(prefix_token_list):]
    word_list = prompt.split()
    token_list = tokenizer.tokenize(prompt)

    assert len(token_list) == new_saliency_score.shape[1] == new_saliency_score.shape[2]

    # Create a mask for each word indicating the corresponding tokens
    word_i = 0
    prev_tokens = []
    token_masks = []
    for i, token in enumerate(token_list[:]):
        word = word_list[word_i]
        token = token.strip('▁')
        if "".join(prev_tokens) == word and token in word_list[min(word_i + 1, len(word_list))]:
            prev_tokens = []
            word_i += 1
            word = word_list[word_i]
        token_masks.append(word_i)
        prev_tokens.append(token)
        if token == word:
            prev_tokens = []
            word_i += 1
        # print(f"{i},{word_i},\t{token},{word}")

    # Calculate the maximum for the tokens in each word
    def reduce_scores(scores, axis):
        return np.maximum.reduceat(scores, np.unique(token_masks, return_index=True)[1], axis=axis)

    word_scores = reduce_scores(reduce_scores(new_saliency_score, axis=1), axis=2)

    # return only a subset of the word list
    start_idx = 0
    end_idx = word_list.index('[/INST]')
    indices = slice(start_idx, end_idx)

    return word_scores[:, indices, indices], word_list[indices]


if __name__ == '__main__':
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.books.books_evaluation import cal_acc_when_fp
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm

    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    question_key = 'when_false_premise_question2'
    subject_key = 'subject_title'
    score_key = 'entropy_over_concepts'
    ground_truth_key = 'who_question_ground_truth'
    answer_key = 'when_false_premise_question2_model_answer'
    generation_key = 'generations'

    # dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_multiple_generation.json'
    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_new_fp_when_false_premise_question2_model_answer.json'
    result_dir = f'/home/zhuoran/hongbang/projects/HalluInducing/results/information_flow/books/{model_size}'
    samples = read_json(dataset_file)
    acc = cal_acc_when_fp(samples, key=answer_key)
    print("Acc:", acc)
    if debug:
        samples = samples[:10]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    model.train()

    embed_layername = 'model.embed_tokens'
    # layers = [embed_layername] + [f'model.layers.{i}.self_attn' for i in range(model.config.num_hidden_layers)]
    layers = [f'model.layers.{i}.self_attn' for i in range(model.config.num_hidden_layers)]


    def patch_rep(x, layer, pached_states):
        h = untuple(x)
        if h.shape[1] == 1:
            # Pay close attention when prompt length=1! But I don't think this would happen.
            return x
        pached_states.append(x[1])
        return x


    # scores = []
    # word_scores = []
    # word_lists = []
    results = []
    for idx, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/','').replace(" ",'_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"

        pached_states = []
        question = sample[question_key]
        answer = sample[answer_key]
        orig_prompt = format_question_answer(question, answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
        prompt = generation[:, :generation.shape[-1] - len(tokenizer.tokenize(remove_prompt(orig_prompt)))]

        target_ids = generation.clone()
        target_ids[:, :prompt.shape[-1]] = -100
        with TraceDict(model, layers, retain_output=False,
                       edit_output=partial(patch_rep, pached_states=pached_states)) as ret:
            model_output = model(torch.reshape(generation, (1, -1)).to(model.device), labels=target_ids.to(model.device), output_attentions=True)
        loss = model_output['loss']
        # loss.backward()
        pached_states_grad = torch.autograd.grad(
            outputs=loss,
            inputs=pached_states,
            retain_graph=True
        )
        pached_states_npy = np.stack([s.clone().detach().cpu().numpy() for s in pached_states]).squeeze()
        pached_states_grad_npy = np.stack(
            [s_grad.clone().detach().cpu().numpy() for s_grad in pached_states_grad]).squeeze()
        saliency_score = np.abs((pached_states_npy * pached_states_grad_npy).sum(axis=1) * 100)
        word_saliency_score, word_list = reshape_saliency_score(
            saliency_score[:, 1:, 1:], orig_prompt, tokenizer
        )
        # scores.append(saliency_score)
        # word_scores.append(word_saliency_score)
        # word_lists.append(word_list)
        result = {
            "states": pached_states_npy,
            "states_grad": pached_states_grad_npy,
            "saliency_score": saliency_score,
            "word_saliency_score": word_saliency_score,
            "word_list": word_list,
        }
        np.savez(filename,**result)
