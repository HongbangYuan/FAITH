import os.path

from baukit import Trace, TraceDict
import torch
import numpy as np
import matplotlib.pyplot as plt

llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
llama_prompt_template = llama_sys_prompt + ' {} [/INST] {}'


def remove_subtring(text,signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal,text))
    return text[end_idx:].strip()

def remove_prompt(text):
    return remove_subtring(text,'[/INST]')

def remove_sys_prefix(text):
    return remove_subtring(text, '<</SYS>>')


def format_question_answer(question,answer):
    return llama_prompt_template.format(question,answer)

def get_llama_attention_bau(model, question,answer):
    prompt = format_question_answer(question, answer)
    encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    ATTNS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = encoded_prompt.to(model.device)
        with TraceDict(model, ATTNS, retain_output=True, retain_input=True, clone=True, detach=True) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True, use_cache=False)
        attn_weights = ret[ATTNS[-1]].output[1].squeeze().max(dim=0).values.cpu().numpy()

    return attn_weights


def reshape_attn_weights(orig_attn_weights,question,answer,tokenizer):
    orig_prompt = format_question_answer(question, answer)
    prompt = remove_prompt(remove_sys_prefix(orig_prompt))
    word_list = prompt.split()
    token_list = tokenizer.tokenize(prompt)
    attention_contribution = orig_attn_weights[-len(token_list):,-len(token_list):]
    assert len(token_list) == attention_contribution.shape[0] == attention_contribution.shape[1]

    # Create a mask for each word indicating the corresponding tokens
    word_i = 0
    token_masks = []
    for i, token in enumerate(token_list[:]):
        word = word_list[word_i]
        token = token.strip('▁')
        if token not in word and token in word_list[min(word_i+1,len(word_list))]:
            word_i += 1
            word = word_list[word_i]
        token_masks.append(word_i)
        if token == word:
            word_i += 1
        # print(f"{i},{word_i},\t{token},{word}")

    # Create an array to store the result
    result_array_ = np.zeros((len(word_list), len(token_masks)))
    # Use boolean indexing to select and calculate the maximum for each group
    result_array_[token_masks] = np.maximum(result_array_[token_masks], attention_contribution)

    # 再次压缩
    result_array = np.zeros((len(word_list), len(word_list)))
    result_array[token_masks] = np.maximum(result_array[token_masks], result_array_.T)
    result_array = result_array.transpose()

    # # return only a subset of the word list
    # start_idx = 0
    # end_idx = word_list.index('[/INST]')
    # indices = slice(start_idx,end_idx)
    indices = slice(0,len(word_list))

    return result_array[indices,:],word_list[indices]


def plot_attn_weight(attn_weight,word_list,title,fig_size=(8,8),save_path=None):
    assert attn_weight.shape[0] == attn_weight.shape[1] == len(word_list)
    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    plt.imshow(attn_weight, cmap='Blues', aspect='auto')
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    # plt.xlabel('Words')
    # plt.ylabel('Words')

    # Add a colorbar to show the scale of values
    plt.colorbar()
    y_labels = word_list
    x_labels = word_list
    plt.xticks(np.arange(len(x_labels)),x_labels,rotation=90)
    plt.yticks(np.arange(len(y_labels)), y_labels)

    # Show the plot
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # dataset: film release why false premise questions   key: false_premise_model_answer
    # '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-13b-chat_on_film_release_dataset_evaluation.json'
    # dataset: film release when questions   key: when_question_model_answer
    # '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-13b-chat_on_film_release_dataset_when_question.json'
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping
    from core.evaluation.film_release_evaluation import cal_acc
    from tqdm import tqdm

    print("Hello World!")

    model_size = '13b'
    debug = False
    question_key = 'false_premise_question'
    answer_key = 'false_premise_model_answer'
    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{model_size}-chat_on_film_release_dataset_evaluation.json'
    figs_path = f'/home/zhuoran/hongbang/projects/HalluInducing/results/generation_attention/film/{model_size}'

    samples = read_json(result_file)
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:",model_name)

    acc = cal_acc(samples,key=answer_key)
    print("acc:",acc)

    model,tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    split = 0
    pbar = enumerate(tqdm(samples[split:]) if not debug else tqdm(samples[:5]))
    k = 20
    start = 0
    indices = slice(start,start+k)

    for idx,sample in pbar:
        save_file_name = os.path.join(figs_path,"{}_{}.png".format(str(idx+split),sample["original_title"].replace('/','').replace(' ','_')))
        question = sample[question_key]
        answer = sample[answer_key]
        attn_weights = get_llama_attention_bau(model, question,answer)
        new_attn_weights,word_list = reshape_attn_weights(attn_weights,question,answer,tokenizer)
        new_attn_weights = new_attn_weights[indices,indices]
        word_list = word_list[indices]
        plot_attn_weight(new_attn_weights,word_list,title=question+' '+str(sample["pred"]),save_path=save_file_name)
        # break
    print("Finished Running")
    print("Hello!")
