from core.inference.llama_inferencer import format_question
import torch
import numpy as np
from collections import defaultdict
from transformers import LlamaForCausalLM
from utils.nethook import TraceDict,Trace
from functools import partial
from core.methods.semantic_uncertainty.generate import remove_prompt
from torch.nn import CrossEntropyLoss
from tqdm import trange,tqdm
from core.evaluation.books.books_evaluation import check_who
import re

llama_sys_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""


def remove_subtring(text,signal):
    start_idx = text.find(signal)
    end_idx = start_idx + len(signal)
    if start_idx == -1:
        raise ValueError("{} not found in {}!".format(signal,text))
    return text[end_idx:].strip()


def remove_sys_prefix(text):
    return remove_subtring(text, '<</SYS>>')


def untuple(x):
    return x[0] if isinstance(x, tuple) else x

def get_predictive_entropy_over_concepts(log_likelihods,semantic_set_ids):
    llh_shift = torch.tensor(0.0)
    aggregated_likelihoods = []
    row = torch.tensor(log_likelihods)
    semantic_set_ids_row = torch.tensor(semantic_set_ids)
    for semantic_set_id in torch.unique(semantic_set_ids_row):
        aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
    entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    return entropy.item()

def get_semantic_ids(generations,ground_truth):
    flags = [check_who(ground_truth,g) for g in generations]
    semantic_ids_per_sample = []
    count = 2
    for flag in flags:
        if flag:
            semantic_ids_per_sample.append(1)
        else:
            semantic_ids_per_sample.append(count)
            count = count + 1
    return semantic_ids_per_sample

class Llama2InferencerWithCorruption:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = self.model.config.num_hidden_layers

    def get_embedding(self, inp, states_to_patch):
        model = self.model
        patch_spec = defaultdict(list)
        for t, l in states_to_patch:
            patch_spec[l].append(t)
        emb_layer = 'model.embed_tokens'
        with torch.no_grad(), TraceDict(
                model,
                [emb_layer] + list(patch_spec.keys()),
                retain_output=True,
                clone=True,
        ) as td:
            output = model(
                input_ids=inp["input_ids"].to(model.device),
                attention_mask=inp["attention_mask"].to(model.device),
            )
        embedding_state = td[emb_layer].output.cpu()
        # patched_states = [untuple(td[key].output) for key in patch_spec.keys()]
        patched_states = {
            key:untuple(td[key].output)[0] for key in patch_spec.keys()
        }

        return embedding_state,patched_states

    def cal_loss_from_output(self,output):
        logits = torch.stack(output.scores).squeeze()
        labels = output.sequences[:,-logits.shape[0]:]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def run_with_noise(
            self,
            prompts,
            inp,
            ground_truth,
            embedding_state,
            states_to_patch,
            pached_states,
            num_generations,
            tokens_to_mix,
            noise,
            uniform_noise,
            replace
    ):
        model,tokenizer = self.model,self.tokenizer
        rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
        if uniform_noise:
            prng = lambda *shape: rs.uniform(-1, 1, shape)
        else:
            prng = lambda *shape: rs.randn(*shape)

        patch_spec = defaultdict(list)
        for t, l in states_to_patch:
            patch_spec[l].append(t)

        embed_layername = layername(model, 0, "embed")

        # Define the model-patching rule.
        if isinstance(noise, float):
            noise_fn = lambda x: noise * x
        else:
            noise_fn = noise

        x = embedding_state
        if tokens_to_mix is not None:
            b, e = tokens_to_mix
            noise_data = noise_fn(
                torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
            ).to(x.device)
            if replace:
                x[1:, b:e] = noise_data
            else:
                x[1:, b:e] += noise_data

        batch_size,num_prompt,hidden_dim = embedding_state.shape
        scores = []
        for i in tqdm(range(1,batch_size),leave=False):
            curr_embedding = x[i].view(1,num_prompt,hidden_dim)
            generations = []
            negative_likelihood_losses = []

            def patch_rep(x, layer):
                if untuple(x).shape[1] == 1:
                    # the noise are already in the kv cache and there is no need to intervene!
                    # Pay close attention when prompt length=1! But I don't think this would happen.
                    return x
                if layer == embed_layername:
                    if tokens_to_mix is not None:
                        h = untuple(x)
                        h = curr_embedding.to(h.device)
                    return x
                if layer not in patch_spec:
                    return x
                h = untuple(x)
                for t in patch_spec[layer]:
                    h[:, t] = pached_states[layer][t].to(h.device)
                return x

            for generate_i in range(num_generations):
                # print(generate_i)
                with torch.no_grad(), TraceDict(
                        model,
                        [embed_layername] + list(patch_spec.keys()) ,
                        edit_output=patch_rep
                ) as td:
                    output = model.generate(
                        # inputs_embeds = curr_embedding.to(model.device),
                        input_ids=inp["input_ids"][i,:].view(1,-1).to(model.device),
                        attention_mask=inp["attention_mask"][i,:].view(1,-1).to(model.device),
                        do_sample=True,
                        num_return_sequences=1,
                        num_beams=1,
                        max_new_tokens=256,
                        temperature=0.5,
                        top_p=1.0,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                #     target_ids = output.sequences.clone()
                #     target_ids[:,:inp["input_ids"][i,:].shape[-1]] = -100
                #     model_output = model(output.sequences, labels=target_ids.to(model.device),
                #                          output_hidden_states=False)
                #
                # average_neg_log_likelihood = model_output['loss'].item()
                decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
                generation = remove_prompt(decoded_output[0])
                generations.append(generation)
                transition_scores = model.compute_transition_scores(
                    output.sequences, output.scores, normalize_logits=True
                )
                negative_likelihood_loss = -transition_scores.mean().item()
                negative_likelihood_losses.append(negative_likelihood_loss)

            # calculate the uncertainty measure
            semantic_ids = get_semantic_ids(generations,ground_truth)
            predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(
                negative_likelihood_losses,semantic_ids
            )
            scores.append(predictive_entropy_over_concepts)
        return scores

def layername(model, num, kind=None):
    if isinstance(model, LlamaForCausalLM):
        # kind = 'self_attn' or 'mlp' or None
        if kind == 'embed':
            return 'model.embed_tokens'
        return f'model.layers.{num}{"" if kind is None else "." + kind}'

    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def calculate_hidden_flow(
        inferencer,
        prompt,
        subject,
        base_score,
        ground_truth,
        num_sample=3,
        noise=0.1,
        token_range=None,
        uniform_noise=False,
        replace=False,
        window=10,
        kind=None,
        expect=None,
):
    model, tokenizer = inferencer.model, inferencer.tokenizer
    prompts = [prompt] * (num_sample + 1)
    batch_input = tokenizer(prompts, return_tensors='pt')
    e_range = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(subject.split()))
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")

    low_score = trace_with_patch(
        inferencer, prompts,ground_truth, batch_input, [], e_range, noise=noise, uniform_noise=uniform_noise,replace=replace
    )
    if not kind:
        print("Tracing important states...")
        differences = trace_important_states(
            inferencer,
            inferencer.num_layers,
            prompts,
            batch_input,
            e_range,
            ground_truth,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range
        )
    else:
        print(f"Tracing important states on kind {kind}...")
        differences = trace_important_window(
            inferencer,
            inferencer.num_layers,
            prompts,
            batch_input,
            e_range,
            ground_truth,
            kind=kind,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range
        )

    return dict(
        scores = differences,
        low_score = low_score,
        high_score = base_score,
        input_ids = batch_input["input_ids"][0],
        input_tokens = prompt,
        subject_range = e_range,
        window = window,
        correct_prediction = True,
        kind = kind or "",
    )

def trace_important_window(
        inferencer,
        num_layers,
        prompts,
        batch_input,
        e_range,
        ground_truth,
        kind,
        window=10,
        noise=0.1,
        uniform_noise=False,
        replace=False,
        token_range=None,
):
    model,tokenizer = inferencer.model,inferencer.tokenizer
    prefix_token_list = tokenizer.tokenize(llama_sys_prompt)
    ntoks = batch_input["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(len(prefix_token_list),ntoks)

    for tnum in token_range:
        row = []
        for layer in tqdm(range(num_layers),leave=False):
            layerlist = [
                (tnum, layername(inferencer.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                inferencer,
                prompts,
                ground_truth,
                batch_input,
                layerlist,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(row)
    table = np.array(table)
    return torch.from_numpy(table)

def trace_important_states(
        inferencer,
        num_layers,
        prompts,
        batch_input,
        e_range,
        ground_truth,
        noise=0.1,
        uniform_noise=False,
        replace=False,
        token_range=None,
):
    model,tokenizer = inferencer.model,inferencer.tokenizer
    prefix_token_list = tokenizer.tokenize(llama_sys_prompt)
    ntoks = batch_input["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(len(prefix_token_list),ntoks)
    for tnum in tqdm(token_range):
        row = []
        for layer in tqdm(range(num_layers),leave=False):
            r = trace_with_patch(
                inferencer,
                prompts,
                ground_truth,
                batch_input,
                [(tnum, layername(inferencer.model, layer))],
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(row)
    table = np.array(table)
    return torch.from_numpy(table)


def trace_with_patch(
        inferencer,  # The model and tokenizer
        prompts,
        ground_truth,
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
        num_generations=5,
):
    embedding_state,patched_states = inferencer.get_embedding(
        inp,states_to_patch
    )
    scores = inferencer.run_with_noise(
        prompts,
        inp,
        ground_truth,
        embedding_state,
        states_to_patch,
        patched_states,
        num_generations=num_generations,
        tokens_to_mix=tokens_to_mix,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace
    )
    return scores

def collect_embedding_std(inferencer, subjects):
    model,tokenizer = inferencer.model, inferencer.tokenizer
    alldata = []
    for s in tqdm(subjects):
        batch_input = tokenizer([s], return_tensors='pt').to(model.device)
        with Trace(model, layername(inferencer.model, 0, "embed"),stop=True) as t:
            model(**batch_input)
        alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level



if __name__ == '__main__':
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json

    model_size = '7b'
    debug = False
    question_key = 'when_false_premise_question'
    subject_key = 'subject_title'
    score_key = 'entropy_over_concepts'
    ground_truth_key = 'who_question_ground_truth'

    dataset_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_base_entropy.json"
    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/books'
    samples = read_json(dataset_file)
    samples = samples if not debug else samples[:2]
    print(f"Load dataset from {dataset_file}")

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    inferencer = Llama2InferencerWithCorruption(model, tokenizer)

    print("Calculating subject embedding noise level...")
    noise_level = 3 * collect_embedding_std(inferencer,[sample[subject_key] for sample in samples])
    print(f"Using noise level {noise_level}")

    for i,sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/','').replace(" ",'_')
        for kind in ["mlp", "self_attn",None]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/{i}_{sample_name}{kind_suffix}.npz"
            print(f"Preparing filename {filename}...")
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    inferencer,
                    format_question(sample[question_key], instruction=""),
                    sample[subject_key],
                    sample[score_key],
                    sample[ground_truth_key],
                    kind=kind,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                np.savez(filename, **numpy_result)
                print(f"Saving result to {filename}!")
            else:
                numpy_result = np.load(filename, allow_pickle=True)

    print("Finished Running!")
