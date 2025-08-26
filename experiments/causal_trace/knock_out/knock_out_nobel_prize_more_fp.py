import random

def generate_random_positions(num_positions, seed=None):
    random.seed(seed)
    positions = []
    for _ in range(num_positions):
        x = random.randint(0, 8)
        y = random.randint(0, 30)
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
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--question_number',type=int,default=1,choices=[1,2,3,4])
    parser.add_argument('--batch_size',type=int,default=2)
    args = parser.parse_args()

    question_number = args.question_number
    batch_size = args.batch_size
    question_key = 'when_fp_question'+str(question_number)
    if question_number == 1:
        question_key = 'when_fp_question'
    output_key = f'knock_out_question_{question_number}'
    print("question_key:",question_key)
    print("output_key:",output_key)

    use_docker = args.use_docker
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/knockout/NobelPrize/more_fps/all_pos'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")

    model_size = f'{args.model_size}b'
    debug = args.debug

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
    # heads_pos = [(6,24)]
    # print("Random positions:",heads_pos)

    if model_size == '7b':
        num_to_heads_pos = {
            1: [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)],
            2: [(2, 2), (1, 22), (8, 18), (1, 15), (11, 16)],
            3: [(1, 22), (2, 2), (1, 15), (9, 20), (11, 6)],
            4: [(2, 2), (1, 22), (1, 15), (13, 6), (3, 19)],
            # 1: [(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)],
            # 2: [(2, 2), (1, 22), (5, 15), (1, 15), (9, 10)],
            # 3: [(2, 2), (1, 22), (1, 15), (5, 15), (10, 25)],
            # 4: [(2, 2), (1, 22), (1, 15), (5, 15), (3, 19)],
        }
        orig_heads_pos = num_to_heads_pos[question_number]
    else:
        num_to_heads_pos = {
            # 1: [(0, 13), (18, 2), (2, 31), (1, 28), (15, 22), (32, 20), (34, 35), (5, 8), (10, 11), (10, 21), (11, 24), (12, 20), (2, 7), (36, 11), (6, 17)],
            # 2: [(2, 31), (10, 11), (18, 2), (1, 28), (2, 7), (3, 38), (8, 14), (10, 30), (15, 22), (6, 17)],
            # 3: [(2, 31), (3, 38), (18, 2), (15, 22), (6, 17), (10, 11), (11, 24), (11, 29), (8, 14), (1, 18), (11, 34), (2, 10), (4, 38), (5, 8)],
            # 4: [(2, 31), (2, 7), (0, 8), (8, 14), (0, 13), (1, 28), (3, 38), (4, 38), (1, 18), (9, 4), (10, 11), (9, 28), (1, 29), (10, 31), (18, 2)],
            # 1: [(0, 13), (18, 2), (10, 11), (8, 14), (10, 30), (34, 35), (15, 22), (2, 31), (3, 38), (12, 20), (2, 7), (9, 28), (11, 6), (11, 24), (35, 2)],
            # 2: [(10, 11), (2, 31), (3, 38), (15, 22), (18, 2), (8, 14), (4, 38), (10, 30), (1, 28), (12, 20), (2, 7), (9, 4), (5, 8), (6, 17), (10, 20)],
            # 3: [(3, 38), (18, 2), (2, 31), (11, 24), (15, 22), (4, 38), (8, 14), (2, 7), (6, 17), (1, 18), (10, 30), (10, 11), (5, 8), (11, 29), (11, 34)],
            # 4: [(2, 31), (2, 7), (8, 14), (0, 8), (3, 38), (1, 28), (18, 2), (4, 38), (0, 13), (9, 28), (9, 4), (1, 18), (10, 31), (15, 22), (10, 11)],
            1: [(0, 13), (18, 2), (2, 31), (1, 28), (15, 22), (32, 20), (34, 35), (5, 8), (10, 11), (10, 21), (11, 24), (12, 20), (2, 7), (36, 11), (6, 17), (7, 28), (8, 14), (9, 5)],
            2: [(2, 31), (10, 11), (18, 2), (1, 28), (2, 7), (3, 38), (8, 14), (10, 30), (15, 22), (6, 17)],
            3: [(2, 31), (3, 38), (18, 2), (15, 22), (6, 17), (10, 11), (11, 24), (11, 29), (8, 14), (1, 18), (11, 34), (2, 10), (4, 38), (5, 8)],
            4: [(2, 31), (2, 7), (0, 8), (8, 14), (0, 13), (1, 28), (3, 38), (4, 38), (1, 18), (9, 4), (10, 11), (9, 28), (1, 29), (10, 31), (18, 2), (2, 10), (3, 37), (4, 25), (1, 25), (1, 30)],
        }
        orig_heads_pos = num_to_heads_pos[question_number]
    heads_pos = orig_heads_pos
    print("Selected heads pos:", heads_pos)
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())

    # tag = "".join([str(i) for i in heads_pos])
    tag = str(args.question_number)
    result_file = f'{result_dir}/{model_name}_on_nobel_prize_knock_out_heads_{tag}_fix_pos_knock_all_pos_top5_threshold0.1.json'
    print(f"Result will be saved in file {result_file}")

    curr_dataset = CustomDataset(samples, question_key)
    curr_data_loader = DataLoader(curr_dataset, batch_size=batch_size)
    pbar = tqdm(curr_data_loader)
    print(f"Processing a total of {len(samples)} samples...")
    i = 0
    for batch in pbar:
        orig_prompts = list(map(lambda x: format_question(x, ""), batch[question_key]))
        batch_input = tokenizer(orig_prompts, padding=True, return_tensors="pt").to(model.device)

        # false_year = str(sample["awardYear"] + 1)
        # orig_prompts = [format_question(question, "")]
        # batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
        # batch_input = {
        #     key: value.to(model.device) for key, value in batch_input.items()
        # }
        # _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question.split()))
        # pos = end_of_question - 2  # the previous token position before the question mark.

        positions = []
        for idx,sample in enumerate(samples[i:i+batch_size]):
            false_year = str(sample["awardYear"] + 1)
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
                # h[:,pos,dim_start:dim_end] = torch.zeros_like(h[:,pos,dim_start:dim_end])
                for pos in positions:
                    b_pos,query_start,query_end = pos
                    # b_pos,query_pos = pos
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
