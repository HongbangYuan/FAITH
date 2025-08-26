
import random

def generate_random_positions(num_positions, seed=None):
    random.seed(seed)
    positions = []
    for _ in range(num_positions):
        x = random.randint(0, 8)
        y = random.randint(0, 39)
        positions.append((x, y))
    return positions


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    from collections import defaultdict
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    import argparse
    import torch
    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices, select
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from core.inference.llama_inferencer import format_question
    from core.methods.information_flow.saliency_score import remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from collections import defaultdict
    from core.methods.causal_trace.casual_trace import find_token_range, untuple
    import os
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/knockout_generation/NobelPrize'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")

    model_size = f'{args.model_size}b'
    debug = args.debug
    question_key = 'when_fp_question'

    samples = load_nobel_prize_only_fp(model_size=args.model_size,use_docker=use_docker)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)


    generation_kwargs = {
        "max_length": 256,
        "num_beams": 5,
        "do_sample": False,
    }
    print("generation kwargs:", generation_kwargs)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers
    # heads_pos = generate_random_positions(5,seed=0)
    # print("Random positions")
    if model_size == '7b':
        # heads_pos = [(1,15),(2,2),(1,22),(5,15),(8,18)] # Movie attn head 7b
        # orig_heads_pos = [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)]  # Nobel Prize attn head7b
        # orig_heads_pos = [(9, 10), (5, 15), (9, 30), (2, 2), (1, 22)] # Nobel Prize attn head7b selected with uncertainty
        # orig_heads_pos = [(9, 10), (9, 30), (5, 15), (2, 2), (9, 18)] # Nobel Prize attn head7b selected with uncertainty threshold=0.01
        orig_heads_pos = [(9, 10), (7, 22), (5, 15), (9, 2), (12, 21)]  # Nobel Prize attn head7b selected with uncertainty threshold=0.1 with knockout
    else:
        # heads_pos = [(8, 14), (2, 31), (10, 11), (12, 38), (2, 7)] # Movie attn 13b
        # orig_heads_pos = [(0, 13), (18, 2), (2, 31), (1, 28), (15, 22)]  # Nobel Prize attn head13b
        # orig_heads_pos = [(0, 13), (10, 11), (18, 2), (10, 30), (3, 38)] # Nobel Prize attn head13b selected with uncertainty
        orig_heads_pos = [(0, 13), (10, 11), (10, 30), (11, 24), (2, 31)] # Nobel Prize attn head13b selected with uncertainty threshold=0.01


    # order = [5,4,3,1,2]
    order = None
    if order is not None:
        heads_pos = [orig_heads_pos[idx-1] for idx in order]
        order_tag = "_" + "".join([str(i) for i in order]) if order is not None else ""
    else:
        heads_pos = orig_heads_pos
        order_tag = ""

    print("Selected heads pos:", heads_pos)
    result_file = f'{result_dir}/{model_name}_on_nobel_prize_knock_out_heads{order_tag}.json'
    print(f"Result will be saved in file {result_file}")

    for k in range(1,len(heads_pos)+1):
    # for k in [3,4,5]:
        curr_heads_pos = heads_pos[:k]
        print("Current heads pos:",curr_heads_pos)

        layer_to_head = defaultdict(list)
        for elem in curr_heads_pos:
            layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
        layers = list(layer_to_head.keys())

        pbar = tqdm(samples)
        print(f"Processing a total of {len(samples)} samples...")
        for idx, sample in enumerate(pbar):
            question = sample[question_key]
            orig_prompts = [format_question(question, "")]
            batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
            batch_input = {
                key: value.to(model.device) for key, value in batch_input.items()
            }
            _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question.split()))
            pos = end_of_question - 2  # the previous token position before the question mark.


            def intervene_head(x, layer):
                h = untuple(x)
                if untuple(x).shape[1] == 1:
                    # Pay close attention when prompt length=1! But I don't think this would happen.
                    return x
                heads = layer_to_head[layer]
                for head in heads:
                    dim_start = head * head_dim
                    dim_end = (head + 1) * head_dim
                    # h[:,pos,dim_start:dim_end] = torch.zeros_like(h[:,pos,dim_start:dim_end])
                    h[:, pos, dim_start:dim_end] = 0
                return x


            with torch.no_grad(), TraceDict(
                    model,
                    layers,
                    edit_input=intervene_head
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
            sample[f"knockout_result_{k}"] = answers[0]
        write_to_json(samples, result_file, default=str)
        print(f"Finished running with knocking out {k} attention heads!")
        print("--------------------------------------------------------")

    print("Writing result to {}!".format(result_file))
    print("Finished Running!")




