import random
import os
import numpy as np
from collections import Counter

def generate_random_positions(num_positions, seed=0):
    random.seed(seed)
    positions = []
    for _ in range(num_positions):
        x = random.randint(0, 8)
        y = random.randint(0, 30)
        positions.append((x, y))
    return positions

def is_file(filepath):
    return os.path.isfile(filepath)

def get_files_except_dicts(directory):
    files = [f for f in os.listdir(directory) if is_file(os.path.join(directory, f))]
    return files


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


def f(x):
    return int(x.split('-')[0]), int(x.split('-')[1])

def get_top_k_attn_head(path_patch_result_dir,k=5,threshold=0.1):
    files = get_files_except_dicts(path_patch_result_dir)
    # print(f"Analyze a total of {len(files)} samples.")
    count = 0
    score_threshold = threshold

    important_heads = []
    for file in files:
        file_path = os.path.join(path_patch_result_dir, file)
        np_result = dict(np.load(file_path, allow_pickle=True))
        # difference score: (num_layers,num_heads)
        score = np_result["differences"]
        if score.max() < score_threshold:
            count += 1
            continue
        top_ks = top_k_elements(score, k=k)
        top_ks = [elem for elem in top_ks if elem[-1] > score_threshold]
        # plot_path_patch(score)
        for elem in top_ks:
            important_heads.append(f"{elem[0]}-{elem[1]}")

    invalid_rate = count / len(files)
    heads = sort_elements_by_frequency(important_heads)

    selected_heads = []
    # print(f"Top heads and frequencies under threshold {score_threshold}:")
    # print(heads[:min(k, len(heads))])
    for i in range(min(k, len(heads))):
        selected_heads.append(f(heads[i][0]))

    return selected_heads, heads[:min(k, len(heads))], invalid_rate


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from dataset.ToyDataset.Movies.load_movies import load_wiki_movie_only_fp
    from collections import defaultdict
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    import argparse
    import torch
    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices, select, CustomDataset
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from core.inference.llama_inferencer import format_question
    from core.methods.information_flow.saliency_score import remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from collections import defaultdict
    from core.methods.causal_trace.casual_trace import find_token_range, untuple
    import os
    from torch.utils.data import DataLoader
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=13, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--question_number',type=int,default=1,choices=[1,2,3,4])
    parser.add_argument('--top_k',type=int,default=15)
    parser.add_argument('--threshold',type=float,default=0.1)
    parser.add_argument('--batch_size',type=int,default=2)
    args = parser.parse_args()

    batch_size = args.batch_size
    question_number = args.question_number
    use_docker = args.use_docker
    debug = args.debug
    threshold = args.threshold
    top_k = args.top_k
    subject_key = 'movie'
    question_key = f'fp_question_{question_number}'
    output_key = f'knock_out_question_{question_number}'
    print("question_key:",question_key)
    print("output_key:",output_key)


    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/knockout/Movies/more_fps/within_knowledge'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")

    model_size = f'{args.model_size}b'
    debug = args.debug

    samples = load_wiki_movie_only_fp(model_size=args.model_size,use_docker=args.use_docker)
    # construct attention heads with each question
    print(f"Select {top_k} heads for all questions with threshold={threshold} ...")
    num_to_heads_pos = {}
    for q in range(1,4+1):
        print(f"--------------q={q}------------------")
        path_patch_result_dir = f'{base_dir}/results/causal_trace/path_patch/Movies/more_fps/head_contributions_{model_size}/question_{q}'
        if not os.path.exists(path_patch_result_dir):
            assert (f"Path path result dir {path_patch_result_dir} not found!")
        heads_pos, selected_heads_with_frequency, invalid_rate = get_top_k_attn_head(path_patch_result_dir,k=top_k,threshold=threshold)
        print("Selected heads:",selected_heads_with_frequency)
        print("Invalid Rate:",invalid_rate)
        num_to_heads_pos[q] = heads_pos

    generation_kwargs = {
        "max_length": 256,
        "num_beams": 5,
        "do_sample": False,
    }
    print("generation kwargs:", generation_kwargs)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers

    # for q in range(1,4+1):
    #     if q != question_number:
    q = 4
    print(f"------------------------------------Select Heads From Q{q}--------------------------")
    # select knock out heads
    heads_pos = num_to_heads_pos[q]
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())

    tag = f'Q_{question_number}_with_head_from_Q{q}_threshold_{str(threshold)}_topk_{str(top_k)}'
    result_file = f'{result_dir}/{model_name}_on_wiki_movie_{tag}.json'
    print(f"Result will be saved in file {result_file}")

    curr_dataset = CustomDataset(samples, question_key)
    curr_data_loader = DataLoader(curr_dataset, batch_size=batch_size)
    pbar = tqdm(curr_data_loader)
    print(f"Processing a total of {len(samples)} samples...")
    i = 0
    for batch in pbar:
        orig_prompts = list(map(lambda x: format_question(x, ""), batch[question_key]))
        batch_input = tokenizer(orig_prompts, padding=True, return_tensors="pt").to(model.device)

        positions = []
        for idx,sample in enumerate(samples[i:i+batch_size]):
            false_year = str(min(sample["time"]) - 1)
            start_of_false_premise, end_of_false_premise = find_token_range(tokenizer, batch_input["input_ids"][idx], "".join(false_year.split()))
            positions.append([idx,start_of_false_premise,end_of_false_premise-1])
            # positions.append([idx,end_of_false_premise-1])
        positions = torch.tensor(positions).to(model.device)

        def intervene_head(x, layer,positions,num_beams):
            h = untuple(x)
            if untuple(x).shape[1] == 1:
                # Pay close attention when prompt length=1! But I don't think this would happen.
                return x
            heads = layer_to_head[layer]
            for head in heads:
                dim_start = head * head_dim
                dim_end = (head + 1) * head_dim
                for pos in positions:
                    # b_pos,query_pos = pos
                    b_pos,query_start,query_end = pos
                    h[b_pos*num_beams:(b_pos+1)*num_beams,query_start:query_end+1,dim_start:dim_end] = 0
            return x


        with torch.no_grad(), TraceDict(
                model,
                layers,
                edit_input=partial(intervene_head,positions=positions,num_beams=generation_kwargs["num_beams"])
        ):
            output = model.generate(
                input_ids=batch_input["input_ids"],
                attention_mask=batch_input["attention_mask"],
                **generation_kwargs,
            )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        answers = [
            remove_prompt(decoded_output[idx]) for idx in range(len(orig_prompts))
        ]
        for idx, sample in enumerate(samples[i: i + batch_size]):
            sample[output_key] = answers[idx]
        i += batch_size
        write_to_json(samples[:i], result_file, default=str)

    print("Writing result to {}!".format(result_file))
    print("Finished Running!")
