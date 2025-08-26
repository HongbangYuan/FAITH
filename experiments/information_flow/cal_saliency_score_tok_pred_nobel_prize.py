
if __name__ == '__main__':
    import torch
    from torch.nn import CrossEntropyLoss
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, select, get_model_name_mapping, load_llama_tokenizer
    from utils.nethook import TraceDict
    from tqdm import tqdm
    # from core.methods.information_flow.saliency_score import format_question_answer,reshape_saliency_score,remove_prompt
    from experiments.information_flow.cal_saliency_score_nobel_prize import reshape_saliency_score ,remove_prompt \
        ,format_question_answer ,untuple
    from experiments.causal_trace.path_patch.single_tok_pred_nobel_prize import format_answer_from_sample
    from functools import partial
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    args = parser.parse_args()

    model_size = f'{args.model_size}b'
    debug = args.debug
    use_docker = args.use_docker

    subject_key = 'name'
    answer_key = 'when_fp_question_model_answer'
    question_key = 'when_fp_question'
    fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'
    truncate_max_length = 256

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/information_flow/NobelPrize/tok_pred/{model_size}'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result path {result_figs_dir}...")

    samples = load_nobel_prize_only_fp(model_size=args.model_size, use_docker=use_docker)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    model.train()
    # tokenizer = load_llama_tokenizer(model_name_or_path)
    loss_fct = CrossEntropyLoss()

    embed_layername = 'model.embed_tokens'
    # layers = [embed_layername] + [f'model.layers.{i}.self_attn' for i in range(model.config.num_hidden_layers)]
    layers = [f'model.layers.{i}.self_attn' for i in range(model.config.num_hidden_layers)]


    def patch_rep(x, layer, pached_states):
        h = untuple(x)
        if h.shape[1] == 1:
            # Pay close attention when prompt length=1! But I don't think this would happen.
            return x
        pached_states.append(x[1])
        return x


    results = []
    for idx, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/' ,'').replace(" " ,'_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"

        question = sample[question_key]
        # question = f'For what specific contribution was {sample["name"]} awarded {sample["categoryFullName"]} in XXXX?'
        uncompleted_answer = format_answer_from_sample(sample)
        # print("Debug Usage")
        false_year = str(sample["awardYear"] + 1)
        true_year = str(sample["awardYear"])
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
        orig_prompt = format_question_answer(question, uncompleted_answer)
        batch_input = tokenizer(orig_prompt, return_tensors="pt")["input_ids"].to(model.device)

        pached_states = []
        with TraceDict(
                model,
                layers,
                retain_output=False,
                edit_output=partial(patch_rep, pached_states=pached_states)
        ) as ret:
            out = model(
                batch_input,
                output_attentions=True,
            ) # out["logits"] = (1,32000)
            loss = loss_fct(out["logits"][0,-1] ,ground_truth_token_id.to(model.device))

        # loss.backward()
        pached_states_grad = torch.autograd.grad(
            outputs=loss,
            inputs=pached_states,
            retain_graph=True
        )
        pached_states_npy = np.stack([s.clone().detach().cpu().numpy() for s in pached_states]).squeeze()
        pached_states_grad_npy = np.stack(
            [s_grad.clone().detach().cpu().numpy() for s_grad in pached_states_grad]).squeeze()
        saliency_score = np.abs((pached_states_npy * pached_states_grad_npy).sum(axis=1) * 100)
        word_saliency_score, word_list = reshape_saliency_score(
            saliency_score[:, 1:, 1:], orig_prompt, tokenizer
        )
        # scores.append(saliency_score)
        # word_scores.append(word_saliency_score)
        # word_lists.append(word_list)
        result = {
            # "states": pached_states_npy,
            # "states_grad": pached_states_grad_npy,
            "saliency_score": saliency_score,
            "word_saliency_score": word_saliency_score,
            "word_list": word_list,
        }
        np.savez(filename ,**result)


    # # debug
    # model_name = "llama2-{}-chat".format(model_size)
    # model_name_mapping = get_model_name_mapping(use_docker)
    # model_name_or_path = model_name_mapping[model_name]
    # print("Model name:", model_name)
    # tokenizer = load_llama_tokenizer(model_name_or_path)
    #
    # # saliency_score = np.random.randn(40,256,256)
    # truncated_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>For what specific contribution was Emil von Behring awarded The Nobel Prize in Physiology or Medicinein 1902? [/INST] Hello! I'm here to help you with your question. Emil von Behring was awarded the Nobel Prize in Physiology or Medicine in 1902 for his groundbreaking work on the development of a diphtheria antitoxin. This  asdf asd asdf{}a -/asda\t\t \n\n\n\n\n\n\nachievement marked a significant milestone in the field of immunology and paved the way for the development of vaccines and other life-saving treatments.\n"
    # dim = len(tokenizer.tokenize(truncated_prompt))
    # saliency_score = np.random.randn(40,dim,dim)
    # word_saliency_score, word_list = reshape_saliency_score(
    #     saliency_score[:, :, :], truncated_prompt, tokenizer
    # )
    #
