import matplotlib.pyplot as plt
import nltk
import string
import os

llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""
llama_prompt_template = llama_sys_prompt + ' {} [/INST] {}'


def format_question_answer(question,answer):
    return llama_prompt_template.format(question,answer)

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



def reshape_attention(orig_attention_contribution,orig_prompt,tokenizer):
    ori_token_list = tokenizer.tokenize(orig_prompt)
    prompt = remove_sys_prefix(orig_prompt)
    prefix_token_list = tokenizer.tokenize(llama_sys_prompt)
    attention_contribution = orig_attention_contribution[len(prefix_token_list):,:]
    word_list = prompt.split()
    token_list = tokenizer.tokenize(prompt)

    assert len(token_list) == attention_contribution.shape[0]

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
    result_array = np.zeros((len(word_list), attention_contribution.shape[1]))
    # Use boolean indexing to select and calculate the maximum for each group
    result_array[token_masks] = np.maximum(result_array[token_masks], attention_contribution)

    # return only a subset of the word list
    start_idx = 0
    end_idx = word_list.index('[/INST]')
    indices = slice(start_idx,end_idx)

    return result_array[indices,:],word_list[indices]

def plot_attention(word_attention,word_list,title,fig_size=(8,8),save_path=None):
    assert word_attention.shape[0] == len(word_list)

    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    plt.imshow(word_attention, cmap='Blues', aspect='auto',vmin=4,vmax=14)
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    plt.xlabel('Layers')
    plt.ylabel('Tokens')

    # Add a colorbar to show the scale of values
    plt.colorbar()
    y_labels = word_list
    plt.yticks(np.arange(len(y_labels)), y_labels)

    # Show the plot
    plt.title(title)
    # plt.show()
    if save_path:
        plt.savefig(save_path)



if __name__ == '__main__':
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    from tqdm import tqdm
    import json
    import numpy as np
    from utils import read_json
    from baukit import TraceDict

    load_dotenv()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from dataset.ToyDataset.Movies.load_movies import load_movies,load_film_release
    from utils import select, CustomDataset,write_to_json,read_json,model_name_mapping,load_llama_model_and_tokenizer

    # select some arguments

    # question = "Is there a movie that was directed by Steven Spielberg and that won Academy Award for best Film Editing?"
    # question = "Tell me the year the basketball player Andrew Wiggins was born in."
    # question = "Tell me the year the basketball player Michael Jordan was born in."
    # answer = "The player was born in"
    batch_size = 8
    # model_size = '7b'
    model_size = '13b'
    key = 'false_premise_question'
    output_key = "false_premise_model_answer"
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/attention/film_release_figs/{}/'.format(model_size)
    print("Model Name:",model_name_or_path)
    print("Result File:",result_path)

    samples = load_film_release()
    data_source = "/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{}-chat_on_film_release_dataset_evaluation.json".format(model_size)
    film_release_dataset = read_json(data_source)
    print("data_source:",data_source)
    # film_release_dataset = CustomDataset(samples, key)
    # Initialize the DataLoader with your dataset
    # data_loader = DataLoader(film_release_dataset, batch_size=batch_size)

    # load models
    model,tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    split = 0
    for idx,sample in enumerate(tqdm(film_release_dataset[split:])):
        question = sample[key]
        answer = sample[output_key]
        save_file_name = os.path.join(result_path,"{}_{}.png".format(str(idx+split),sample["original_title"].replace('/','')))

        prompt = format_question_answer(question,answer)
        inputs = tokenizer(prompt,return_tensors='pt').to(model.device)
        HEADS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
        with TraceDict(model, HEADS, retain_output=True, retain_input=True, clone=True, detach=True) as ret:
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=0, output_attentions=True, use_cache=False)
        # result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        attention_contribution = np.array(
            [ret[head].output[0].squeeze().detach().cpu().norm(dim=-1).tolist() for head in HEADS]).T

        word_attention,word_list = reshape_attention(attention_contribution,result,tokenizer)
        plot_attention(word_attention,word_list,title=question+str(sample["pred"]),save_path=save_file_name)
    print("Finished Running!")
