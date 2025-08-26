from core.methods.causal_trace.casual_trace import find_token_range
from core.methods.causal_trace.causal_trace_tok_pred import decode_tokens
import matplotlib.pyplot as plt
from core.methods.causal_trace.path_patch import plot_path_patch, untuple
from torch.utils.data import DataLoader,Dataset

class PathPatchDataset(Dataset):
    def __init__(self,batch_input,num_hidden_layers,num_heads):
        self.batch_input = batch_input
        self.input_ids = batch_input["input_ids"]
        self.attention_mask = batch_input["attention_mask"]
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads

    def __len__(self):
        return self.num_hidden_layers // 2 * self.num_heads

    def __getitem__(self, index):
        return [
            self.input_ids,
            self.attention_mask,
            index // self.num_heads,
            index % self.num_heads
        ]

def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    # p, preds = torch.max(probs, dim=1)
    return probs


def calculate_path_patch(
        model,
        tokenizer,
        question_reference,
        question_counter_factual,
        false_year,
        uncompleted_answer,
        batch_size = 1,
        ground_truth_token=None,
        ground_truth_token_id=None,
        threshold=0.1
):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_hidden_layers = model.config.num_hidden_layers

    orig_prompts = [
        format_question_answer(question_reference, uncompleted_answer),
        format_question_answer(question_counter_factual, uncompleted_answer),
        format_question_answer(question_reference, uncompleted_answer),
    ]
    batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
    batch_input = {
        key: value.to(model.device) for key, value in batch_input.items()
    }
    with torch.no_grad():
        probs = predict_from_input(model, batch_input)
        p, preds = torch.max(probs, dim=1)
        answers_t, base_scores = [[d[0], d[1]] for d in [preds, p]]
    [predicted_token, counter_factual_token] = decode_tokens(tokenizer, answers_t)
    if predicted_token == ground_truth_token:
        print(f"Predicted Token {predicted_token} = Ground Truth Token {ground_truth_token}!")
        return None
        # raise ValueError(f"Predicted Token {predicted_token} = Ground Truth Token {ground_truth_token}!")
    if counter_factual_token != ground_truth_token:
        init_ground_truth_prob = probs[0, ground_truth_token_id].item()
        counter_factual_ground_truth_prob = probs[1, ground_truth_token_id].item()
        if counter_factual_ground_truth_prob - init_ground_truth_prob < threshold:
            print(f"Token increase only from {init_ground_truth_prob} to {counter_factual_ground_truth_prob}.")
            return None

    _,end_of_false_premise = find_token_range(tokenizer,batch_input["input_ids"][0],"".join(false_year.split()))
    _, end_of_knocked_false_premise = find_token_range(tokenizer,batch_input["input_ids"][1],"XXXX")
    # _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question_reference.split()))
    # pos = end_of_question - 2  # the previous token position before the question mark.
    end_of_false_premise_pos = end_of_false_premise - 1
    end_of_knocked_false_premise_pos = end_of_knocked_false_premise - 1
    layers = [layername(model, L, 'self_attn.o_proj') for L in range(num_hidden_layers)]

    dataset = PathPatchDataset(batch_input,num_hidden_layers,num_heads)
    data_loader = DataLoader(dataset,batch_size=batch_size)
    results = []
    num_inside_batch = len(orig_prompts)
    for piece_of_data in tqdm(data_loader):
        input_ids,attention_mask,selected_layers,selected_heads = piece_of_data
        selected_layers = [layername(model, L.item(), 'self_attn.o_proj') for L in selected_layers]

        def patch_rep(x, layer):
            # before the o_proj layer:(batch_size,query_length,num_head * head_dim)
            h = untuple(x)
            # keeping all the heads frozen to there activations on reference data
            for i in range(batch_size):
                h[2 + i*num_inside_batch,end_of_false_premise_pos,:] = h[0 + i*num_inside_batch,end_of_false_premise_pos,:]
                if layer == selected_layers[i]:
                    selected_head = selected_heads[i]
                    dim_start = selected_head * head_dim
                    dim_end = (selected_head + 1) * head_dim
                    h[2 + i*num_inside_batch, end_of_false_premise_pos, dim_start:dim_end] = h[1 + i*num_inside_batch, end_of_knocked_false_premise_pos, dim_start:dim_end]
            return x

        with torch.no_grad(), TraceDict(
                model,
                layers,
                edit_input=patch_rep,
        ) as td:
            out = model(
                input_ids=input_ids.view(-1,input_ids.shape[-1]),
                attention_mask=attention_mask.view(-1,attention_mask.shape[-1]),
                output_hidden_states=True,
            )  # logits:(batch_size,query_length,vocab_size)
            for i in range(batch_size):
                probs = torch.softmax(out["logits"][(i) * num_inside_batch : (i+1)*num_inside_batch, -1], dim=1)
                ground_truth_token_t = answers_t[1]
                init_ground_truth_score = probs[0, ground_truth_token_t]
                after_ground_truth_score = probs[2, ground_truth_token_t]
                difference = (after_ground_truth_score - init_ground_truth_score).item()
                results.append(difference)
    differences = np.array(results).reshape(num_hidden_layers//2,num_heads)
    return dict(
        differences=differences,
        predicted_token=predicted_token,
        counter_factual_token=counter_factual_token,
        answers_t=np.array([elem.item() for elem in answers_t]),
        base_scores=np.array([elem.item() for elem in base_scores]),
    )


if __name__ == '__main__':
    import torch
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, select, get_model_name_mapping, load_llama_tokenizer
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from experiments.our_method.wiki_movie.single_tok_pred_wiki_movie import format_answer_from_sample
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--question_number',type=int,default=2,choices=[1,2,3,4])
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    use_docker = args.use_docker
    question_number = args.question_number
    batch_size = args.batch_size

    subject_key = 'movie'
    # answer_key = 'when_fp_question_model_answer'
    question_key = 'fp_question_'+str(question_number)
    # fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'+str(question_number)

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/causal_trace/path_patch/Movies/more_fps/head_contributions_{model_size}/question_{question_number}_batch'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        print(f"Creating the result path {result_figs_dir}")
        os.makedirs(result_figs_dir)

    dataset_file = f"{base_dir}/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movie_more_fps_token_answer.json"
    orig_samples = read_json(dataset_file)
    samples = orig_samples
    print(f"A total of {len(samples)} samples.")

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # tokenizer = load_llama_tokenizer(model_name_or_path)

    for i, sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        filename = f"{result_dir}/{i}_{sample_name}.npz"

        if sample[token_result_key][0]:
            continue
        template1 = f"Why was the film {sample['movie']} released in XXXX?"
        template2 = f"What was the film {sample['movie']} released in XXXX describing about?"
        template3 = f"What was the XXXX film {sample['movie']} about?"
        template4 = f"Who are the main characters in the XXXX film {sample['movie']}?"
        question_number_to_counter_factual= {
            1:template1,
            2:template2,
            3:template3,
            4:template4,
        }

        if not os.path.isfile(filename):
            print(f"Processing sample {i} {sample_name}")
            question_reference = sample[question_key]
            question_counter_factual = question_number_to_counter_factual[question_number]
            uncompleted_answer = format_answer_from_sample(sample,args.model_size)
            false_year = str(min(sample["time"]) - 1)
            true_year = str(min(sample["time"]))
            true_token_ids = tokenizer([true_year], return_tensors='pt')["input_ids"][0][1:]
            false_token_ids = tokenizer([false_year], return_tensors='pt')["input_ids"][0][1:]
            prev_common = []
            for true_token_id, false_token_id in zip(true_token_ids, false_token_ids):
                if true_token_id != false_token_id:
                    break
                else:
                    prev_common.append(true_token_id)
            uncompleted_answer += tokenizer.decode(prev_common)
            ground_truth_token_id = true_token_id
            ground_truth_token = tokenizer.decode([ground_truth_token_id])

            np_result = calculate_path_patch(
                model,
                tokenizer,
                question_reference,
                question_counter_factual,
                str(false_year),
                uncompleted_answer,
                batch_size=batch_size,
                ground_truth_token=ground_truth_token,
                ground_truth_token_id=ground_truth_token_id,
            )
            if np_result is not None:
                print(f"Saving to file {filename}")
                np.savez(filename, **np_result)
            else:
                print(f"Skip this sample {i}!")
                continue
        else:
            np_result = np.load(filename, allow_pickle=True)

        # print("Debug Usage")
        pdf_fig_name = f'{i}_{sample_name}.pdf'
        pdf_save_file = f'{result_figs_dir}/{pdf_fig_name}'
        pdf_title = f"{i}_{sample_name}"
        result = dict(np_result)
        plot_path_patch(result["differences"],title=pdf_title,save_path=pdf_save_file)

    print("Finished Running!")
