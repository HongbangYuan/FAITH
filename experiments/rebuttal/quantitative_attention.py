# 定量分析不同的attention head的影响
# 画一些定位得到的attention head的权重示意图；再画一些随机选择的其它attention head的事业图，展示不同的注意力权重图案
import os.path

from utils.nethook import TraceDict,Trace
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
    # prompt = format_question_answer(question, answer)
    prompt = question
    encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    ATTNS = [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        encoded_prompt = encoded_prompt.to(model.device)
        with TraceDict(model, ATTNS, retain_output=True, retain_input=True, clone=True, detach=True) as ret:
            output = model(encoded_prompt, output_hidden_states = True,output_attentions=True, use_cache=False)
        # attn_weights = ret[ATTNS[-1]].output[1].squeeze().max(dim=0).values.cpu().numpy()
        attn_weights_per_head = np.stack([ret[layer].output[1].squeeze().cpu().numpy() for layer in ATTNS])

    return attn_weights_per_head


def reshape_attn_weights(orig_attn_weights,question,movie,false_year,tokenizer):
    # attn_weight:(32,32,19,19) (num_layers,num_heads,seq_length,seq_length)
    # orig_prompt = format_question_answer(question, answer)
    # orig_prompt = question
    # prompt = remove_prompt(remove_sys_prefix(orig_prompt))
    prompt = question
    word_list = prompt.split()
    word_list = ["Why",'was','the','film',movie,'released in',str(false_year),"?"]
    token_list = tokenizer.tokenize(prompt)
    attention_contribution = orig_attn_weights[:,:,-len(token_list):,-len(token_list):]
    assert len(token_list) == attention_contribution.shape[-2] == attention_contribution.shape[-1]
    num_layers,num_heads = orig_attn_weights.shape[0],orig_attn_weights.shape[1]

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

    def reduce_scores(scores, axis):
        return np.maximum.reduceat(scores, np.unique(token_masks, return_index=True)[1], axis=axis)

    word_scores = reduce_scores(reduce_scores(orig_attn_weights, axis=-1), axis=-2)

    return word_scores


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


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def find_token_range(tokenizer, token_array, substring, return_slice=False):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index("".join(substring.split()))
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len("".join(substring.split())):
            tok_end = i + 1
            break
    assert "".join(toks[tok_start:tok_end]) == "".join(substring.split())
    if return_slice:
        return slice(tok_start, tok_end)
    return (tok_start, tok_end)


if __name__ == '__main__':

    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    from tqdm import tqdm
    import argparse


    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()


    model_size = f'{args.model_size}b'
    use_docker = args.use_docker

    debug = False
    question_key = 'fp_question_1'
    answer_key = 'fp_question_1_model_answer'
    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_more_fp_fp_question_4_model_answer.json'
    # figs_path = f'/home/zhuoran/hongbang/projects/HalluInducing/results/generation_attention/film/{model_size}'

    samples = read_json(result_file)
    acc = cal_acc_wiki_movies(samples,key=answer_key)
    print("acc:",acc)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    split = 0
    pbar = enumerate(tqdm(samples[split:]) if not debug else tqdm(samples[:5]))
    k = 20
    start = 0
    indices = slice(start,start+k)

    weights = []
    for idx,sample in pbar:
        # save_file_name = os.path.join(figs_path,"{}_{}.png".format(str(idx+split),sample["original_title"].replace('/','').replace(' ','_')))
        question = sample[question_key]
        answer = sample[answer_key]
        generation = tokenizer(question, return_tensors="pt")["input_ids"][0]
        attn_weights_per_head = get_llama_attention_bau(model, question,answer)
        # film_name_range = find_token_range(tokenizer,generation,sample["movie"],return_slice=True)
        # print("Hello!")
        word_weights = reshape_attn_weights(attn_weights_per_head,question,sample["movie"],sample["false_year"],tokenizer)
        weights.append(word_weights)
        # new_attn_weights = new_attn_weights[indices,indices]
        # word_list = word_list[indices]
        # plot_attn_weight(new_attn_weights,word_list,title=question+' '+str(sample["pred"]),save_path=save_file_name)
        # break
    weights_npy = np.stack(weights).mean(axis=0)
    save_file = "/home/zhuoran/hongbang/projects/HalluInducing/results/rebuttal/weights_llama_7b_on_movies.npy"
    np.save(save_file,weights_npy)
    print("Finished Running")
    print("Hello!")


