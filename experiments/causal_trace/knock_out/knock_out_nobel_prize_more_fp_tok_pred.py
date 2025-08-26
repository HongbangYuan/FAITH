import random

from scipy.spatial import distance
from utils import max_indices
import matplotlib.pyplot as plt
import torch
import re

def format_answer_from_sample(sample):
    # question = sample["when_false_premise_question"]
    # pattern = r"When did (?P<false_author>.*?) write the book (?P<book_name>.*?)\?"
    # match = re.match(pattern, question)
    # if match:
    #     false_author = match.group("false_author")
    #     book_name = match.group("book_name")
    # else:
    #     raise ValueError(f"Book name and author not found in {question}!")


    answer_template = "According to my knowledge, {} won the {} in "
    uncompleted_answer = answer_template.format(sample["name"],sample["categoryFullName"])

    return uncompleted_answer

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
    import torch.nn as nn
    from tqdm import tqdm
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    from collections import defaultdict
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    import argparse
    import torch
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices,select,get_model_name_mapping
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from collections import defaultdict
    from core.methods.causal_trace.casual_trace import find_token_range,untuple
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--question_number',type=int,default=2,choices=[1,2,3,4])
    parser.add_argument('--batch_size',type=int,default=2)
    args = parser.parse_args()

    use_docker = args.use_docker
    # base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    # result_dir = f'{base_dir}/results/causal_trace/knockout/NobelPrize/more_fps'
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    #     print(f"Creating result path {result_dir}...")

    question_number = args.question_number
    batch_size = args.batch_size
    question_key = 'when_fp_question'+str(question_number)
    if question_number == 1:
        question_key = 'when_fp_question'
    output_key = f'knock_out_question_{question_number}'
    token_result_key = f'token_pred_{question_number}'
    ground_truth_key = f'ground_truth_token_{question_number}'
    fp_result_key = 'when_fp_answer_eval'

    print("question_key:",question_key)
    print("output_key:",output_key)

    samples = load_nobel_prize_only_fp(model_size=args.model_size)


    model_size = f'{args.model_size}b'
    debug = args.debug
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
    # heads_pos = generate_random_positions(5,seed=0)
    # print("Random positions")
    num_to_heads_pos = {
        1:[(2, 2), (9, 10), (5, 15), (1, 22), (1, 15)] ,
        2:[(2, 2), (1, 22), (8, 18), (1, 15), (11, 16)],
        3:[(1, 22), (2, 2), (1, 15), (9, 20), (11, 6)],
        4:[(2, 2), (1, 22), (1, 15), (13, 6), (3, 19)],
    }
    if model_size == '7b':
        orig_heads_pos = num_to_heads_pos[question_number]
    else:
        orig_heads_pos = [(0, 13), (18, 2), (2, 31), (1, 28), (15, 22)]  # Nobel Prize attn head13b
    heads_pos = orig_heads_pos
    print("Selected heads pos:", heads_pos)
    layer_to_head = defaultdict(list)
    for elem in heads_pos:
        layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
    layers = list(layer_to_head.keys())



    results = []
    rank_in_false_samples = []
    score_in_false_samples = []
    ground_truth_ranks = []
    predicted_token_ranks = []
    pbar = tqdm(samples)
    true_count = 0
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            question = sample[question_key]
            uncompleted_answer = format_answer_from_sample(sample)
            # print("Debug Usage")
            false_year = str(sample["awardYear"] + 1)
            true_year = str(sample["awardYear"])
            true_token_ids = tokenizer([true_year],return_tensors='pt')["input_ids"][0][1:]
            false_token_ids = tokenizer([false_year],return_tensors='pt')["input_ids"][0][1:]
            prev_common = []
            for true_token_id,false_token_id in zip(true_token_ids,false_token_ids):
                if true_token_id != false_token_id:
                    break
                else:
                    prev_common.append(true_token_id)
            uncompleted_answer += tokenizer.decode(prev_common)
            ground_truth_token_id = true_token_id
            ground_truth_token = tokenizer.decode([ground_truth_token_id])
            orig_prompt = format_question_answer(question, uncompleted_answer)
            batch_input = tokenizer([orig_prompt], return_tensors="pt", padding=True)
            batch_input = {
                key: value.to(model.device) for key, value in batch_input.items()
            }
            _, end_of_false_premise = find_token_range(tokenizer, batch_input["input_ids"][0],
                                                       "".join(false_year.split()))
            pos = end_of_false_premise - 1

            def intervene_head(x,layer):
                h = untuple(x)
                heads = layer_to_head[layer]
                for head in heads:
                    dim_start = head * head_dim
                    dim_end = (head+1) * head_dim
                    # h[:,pos,dim_start:dim_end] = torch.zeros_like(h[:,pos,dim_start:dim_end])
                    h[:,pos,dim_start:dim_end] = 0
                return x


            with torch.no_grad(),TraceDict(
                model,
                layers,
                edit_input=intervene_head
            ):
                out = model(
                    **batch_input,
                    output_hidden_states=True,
                )
                probs = torch.softmax(out["logits"][:, -1], dim=1)
                base_score, answer = torch.max(probs, dim=1)
                predicted_token = tokenizer.decode(answer)
            sample[ground_truth_key] = ground_truth_token
            sample[token_result_key] = (ground_truth_token == predicted_token,predicted_token,base_score.item())
            true_count += (ground_truth_token == predicted_token)
            pbar.set_description(f"{true_count}/{idx+1} acc={true_count/(idx+1):.4f}")
            results.append([ground_truth_token,predicted_token,base_score.item(),sample[fp_result_key]])

    token_predictions = [sample[token_result_key][0] for sample in samples]
    token_acc = sum(token_predictions) / len(results)
    print(f"token_acc:{token_acc:.6f}")
    # analyze_samples = [sample if sample[token_result_key][0] == False else None for sample in samples]
    # analyze_samples = [sample for sample in samples  if sample[token_result_key][0] == False]
    # tok_result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/NobelPrize/llama2-{model_size}-chat_on_nobel_prize_when_fp_question_token_answer.json"
    # write_to_json(samples,tok_result_file)
    # print(f"Writing to tok result file {tok_result_file}")

    # fp_predictions = [sample[fp_result_key] for sample in samples]
    # correlation_coefficient = np.corrcoef(token_predictions, fp_predictions)[0, 1]
    # print(correlation_coefficient)
    # selected_samples = select(samples,token_pred=False)
    # selected_samples = [sample if not(sample[token_result_key][0] == True and sample[fp_result_key] == True)  else None  for sample in samples ]
    # selected_scores = np.array([sample[token_result_key][2].item() for sample in selected_samples])
    # selected_token_preds = sum([sample[token_result_key][1] == str(sample["false_year"])[-1] for sample in selected_samples])/len(selected_samples)
