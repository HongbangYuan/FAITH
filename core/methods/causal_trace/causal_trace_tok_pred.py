from utils.nethook import TraceDict, Trace
from core.methods.causal_trace.casual_trace import find_token_range, decode_tokens, layername
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def make_inputs(tokenizer, prompts, device):
    inp = tokenizer(prompts, return_tensors='pt')
    return dict(
        input_ids=inp["input_ids"].to(device),
        attention_mask=inp["attention_mask"].to(device),
    )


def calculate_hidden_flow(
        model,
        tokenizer,
        prompt,
        subject,
        question,
        num_sample=10,
        noise=0.1,
        token_range=None,
        uniform_noise=False,
        replace=False,
        window=10,
        kind=None,
        expect=None
):
    prompts = [prompt] * (num_sample + 1)
    batch_input = make_inputs(tokenizer, prompts, model.device)
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(model, batch_input)]
    [answer] = decode_tokens(tokenizer, [answer_t])
    if expect is not None and answer != expect:
        return dict(correct_prediction=False)

    e_range = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(subject.split()))
    if token_range == 'question_only':
        token_range = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question.split()))
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")

    low_score = trace_with_patch(
        model, batch_input, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()

    if not kind:
        differences = trace_important_states(
            model,
            model.config.num_hidden_layers,
            batch_input,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            model,
            model.config.num_hidden_layers,
            batch_input,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=batch_input["input_ids"][0],
        input_tokens=decode_tokens(tokenizer, batch_input["input_ids"][0]),
        subject_range=e_range,
        token_range=token_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )



def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    else:
        token_range = range(token_range[0],token_range[1])
    for tnum in tqdm(token_range):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    else:
        token_range = range(token_range[0],token_range[1])
    for tnum in tqdm(token_range):
        row = []
        for layer in tqdm(range(num_layers),leave=False):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)



def trace_with_patch(
        model,  # The model
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def format_answer_from_sample(sample):
    # answer_template = "I apologize, but the film \"{}\" was not released in {}. The film was actually released in {}"
    # answer_template = "I apologize, but the film \"{}\" was actually released in {}"
    # answer_template = "According to my knowledge, the film \"{}\" was released in {}"
    answer_template = "According to my knowledge, the film you mentioned was released in {}"
    film_name = sample["movie"]
    false_year = sample["false_year"]
    true_year = str(sample["time"][0])
    uncompleted_answer = answer_template.format(true_year[:-1])
    ground_truth_token = true_year[-1]
    # return answer_template.format(film_name, false_year, true_year[:-1]),true_year[-1]
    return uncompleted_answer, ground_truth_token


def collect_embedding_std(model, tokenizer, subjects):
    alldata = []
    embed_layer_name = layername(model, 0, "embed")
    for s in tqdm(subjects):
        batch_input = tokenizer([s], return_tensors='pt').to(model.device)
        with Trace(model, embed_layer_name, stop=True) as t:
            model(**batch_input)
        alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def plot_trace_heatmap(trace_result,tokens,fig_size=(8,8),title=None,save_path=None):
    assert len(tokens) == trace_result.shape[0]

    # Increase the size of the figure
    fig = plt.figure(figsize=fig_size)  # Adjust the figure size as needed
    plt.imshow(trace_result, cmap='Blues', aspect='auto')
    # Set labels for x and y axes
    plt.ylabel('Tokens')
    plt.xlabel('Layers')
    curr_layers = [f"layer {i}" for i in range(trace_result.shape[-1])]
    plt.xticks(np.arange(len(curr_layers)), curr_layers,rotation=90)
    # plt.tick_params(axis='x', top=True, labeltop=True,bottom=False,labelbottom=False)
    plt.yticks(np.arange(len(tokens)),tokens)
    # Add a colorbar to show the scale of values
    plt.colorbar()

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    import torch
    import numpy as np
    from functools import partial
    from core.methods.semantic_uncertainty.generate import remove_prompt
    from core.inference.llama_inferencer import format_question
    from core.evaluation.film_release_evaluation import cal_acc_wiki_movies
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices, select
    from utils.nethook import TraceDict
    import torch.nn as nn
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import format_question_answer, remove_prompt

    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    subject_key = 'movie'
    answer_key = 'why_fp_question_model_answer'
    question_key = 'why_fp_question'
    # question_key = 'when_question'
    fp_result_key = 'fp_pred'
    token_result_key = 'token_pred'
    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/Movies'
    result_figs_dir = f'{result_dir}/figs'

    dataset_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    orig_samples = read_json(dataset_file)
    samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    samples = [sample for sample in samples_one_time if
               str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]

    acc = cal_acc_wiki_movies(samples, key=answer_key, output_key=fp_result_key)
    print("Acc:{}".format(acc))
    if debug:
        samples = samples[:5]

    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    # model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    tokenizer = load_llama_tokenizer(model_name_or_path)

    # print("Calculating subject embedding noise level...")
    # noise_level = 3 * collect_embedding_std(model, tokenizer, [sample[subject_key] for sample in samples])
    # print(f"Using noise level {noise_level}")

    for i, sample in enumerate(samples):
        sample_name = sample[subject_key].replace('/', '').replace(" ", '_')
        for kind in ["mlp", "self_attn", None]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/{i}_{sample_name}{kind_suffix}.npz"

            if not os.path.isfile(filename):
                print(f"Preparing filename {filename}...")
                question = sample[question_key]
                uncompleted_answer, ground_truth_token = format_answer_from_sample(sample)
                orig_prompt = format_question_answer(question, uncompleted_answer)

                result = calculate_hidden_flow(
                    model,
                    tokenizer,
                    orig_prompt,
                    sample[subject_key],
                    question,
                    token_range='question_only',
                    noise=noise_level,
                    kind=kind,
                    window=5,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                np.savez(filename, **numpy_result)
            else:
                print(f"Loading from {filename}...")
                numpy_result = np.load(filename, allow_pickle=True)

            plot_result = dict(numpy_result)
            trace_result = plot_result["scores"]
            token_range = plot_result["token_range"]
            tokens = plot_result["input_tokens"][token_range[0]:token_range[1]]
            pdf_title = f"{i}_{sample_name}{kind_suffix}"
            pdf_fig_name = f"{i}_{sample_name}{kind_suffix}.pdf"
            pdf_save_file = f'{result_figs_dir}/{pdf_fig_name}'
            plot_trace_heatmap(trace_result,tokens,fig_size=(8,10),title=pdf_title,save_path=pdf_save_file)
