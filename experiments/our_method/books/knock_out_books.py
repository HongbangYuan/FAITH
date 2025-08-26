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
    from dataset.ToyDataset.Books.load_books import load_books,load_fp_books
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
    import re

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--question_number',type=int,default=1,choices=[1,2,3,4])
    parser.add_argument('--batch_size',type=int,default=2)
    args = parser.parse_args()

    batch_size = args.batch_size
    question_number = args.question_number
    use_docker = args.use_docker
    debug = args.debug
    subject_key = 'subject_title'
    # answer_key = 'when_fp_question_model_answer'
    question_key = f'when_false_premise_question2'
    # fp_result_key = 'when_fp_answer_eval'
    # token_result_key = 'token_pred'
    # ground_truth_key = 'ground_truth_token'
    output_key = f'knock_out_question2'
    print("question_key:",question_key)
    print("output_key:",output_key)


    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/knockout/Books/more_fps'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")

    model_size = f'{args.model_size}b'
    debug = args.debug

    samples = load_fp_books(model_size=args.model_size,use_docker=args.use_docker)[:500]
    print(f"Selected a total of {len(samples)} samples.")

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    # tag = "".join([str(i) for i in heads_pos])
    # tag = str(args.question_number)
    result_file = f'{result_dir}/{model_name}_on_books_knock_out_heads.json'
    print(f"Result will be saved in file {result_file}")

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
        orig_heads_pos = [(1,27),(3,23),(1,24),(2,16),(2,17)]
    else:
        orig_heads_pos = [(1,30),(4,38),(1,28),(2,12),(4,25)]
    heads_pos = orig_heads_pos
    print("Selected heads pos:", heads_pos)
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())


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
            question = sample["when_false_premise_question"]
            pattern = r"When did (?P<false_author>.*?) write the book (?P<book_name>.*?)\?"
            match = re.match(pattern, question)
            if match:
                false_author = match.group("false_author")
            else:
                raise ValueError(f"False author not found in question {question}!")
            _, end_of_false_premise = find_token_range(tokenizer, batch_input["input_ids"][idx], "".join(false_author.split()))
            positions.append([idx,end_of_false_premise-1])
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
                    b_pos,query_pos = pos
                    h[b_pos*num_beams:(b_pos+1)*num_beams,query_pos,dim_start:dim_end] = 0
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
