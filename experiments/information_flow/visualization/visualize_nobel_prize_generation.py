

if __name__ == '__main__':
    import torch
    from torch.nn import CrossEntropyLoss
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, select, get_model_name_mapping, load_llama_tokenizer
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from experiments.information_flow.cal_saliency_score_nobel_prize import reshape_saliency_score ,remove_prompt \
        ,format_question_answer ,untuple
    from experiments.causal_trace.path_patch.single_tok_pred_nobel_prize import format_answer_from_sample
    from experiments.information_flow.information_from_author import find_token_range,plot_contrasive_scores,plot_scores
    from functools import partial
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse
    import matplotlib.pyplot as plt

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
    object_key = 'awardYear'
    answer_key = 'when_fp_question_model_answer'
    question_key = 'when_fp_question'
    fp_result_key = 'when_fp_answer_eval'
    token_result_key = 'token_pred'
    truncate_max_length = 256

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/information_flow/NobelPrize/generation/{model_size}'
    result_figs_dir = f'{result_dir}/figs'
    if not os.path.exists(result_figs_dir):
        os.makedirs(result_figs_dir)
        print(f"Creating result path {result_figs_dir}...")

    samples = load_nobel_prize_only_fp(model_size=args.model_size, use_docker=use_docker)

    model_name = "llama2-{}-chat".format(model_size)
    model_name_mapping = get_model_name_mapping(use_docker)
    model_name_or_path = model_name_mapping[model_name]
    print("Model name:", model_name)
    tokenizer = load_llama_tokenizer(model_name_or_path)


    results = []
    true_sample_scores_subject = []
    true_sample_scores_false_object = []
    false_sample_scores_subject = []
    false_sample_scores_false_object = []
    true_sample_scores_other = []
    false_sample_scores_other = []
    available_count = 0
    for idx, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/' ,'').replace(" " ,'_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"
        np_result = dict(np.load(filename,allow_pickle=True))
        # # Squeeze the original file as it's two large
        # for key in ['states','states_grad']:
        #     if key in np_result:
        #         del np_result[key]
        # np.savez(filename,**np_result)

        if not os.path.isfile(filename):
            continue
        try:
            result = dict(np.load(filename,allow_pickle=True))
        except:
            continue
        saliency_score = result["saliency_score"]
        available_count += 1

        question = sample[question_key]
        answer = sample[answer_key]
        orig_prompt = format_question_answer(question, answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
        prompt = generation[:, :generation.shape[-1] - len(tokenizer.tokenize(remove_prompt(orig_prompt)))]
        truncated_generation = generation[:, :truncate_max_length][0]
        truncated_prompt = tokenizer.decode(truncated_generation[1:])

        subject_range = find_token_range(tokenizer,truncated_generation,sample[subject_key])
        false_year = sample[object_key] + 1
        false_object_range = find_token_range(tokenizer,truncated_generation,str(false_year))
        _, end_of_question = find_token_range(tokenizer, truncated_generation, "[/INST]")

        info_from_subject_range = saliency_score[:,end_of_question:,subject_range[0]:subject_range[-1]].mean(axis=(-1,-2))
        info_from_false_object_range = saliency_score[:,end_of_question:,false_object_range[0]:false_object_range[-1]].mean(axis=(-1,-2))
        other_range = torch.ones(saliency_score.shape[-1],dtype=torch.bool)
        other_range[subject_range[0]:subject_range[-1]] = False
        other_range[false_object_range[0]:false_object_range[-1]] = False
        info_from_other = saliency_score[:,end_of_question:,false_object_range[0]:false_object_range[-1]].mean(axis=(-1,-2))
        if sample[fp_result_key]:
            true_sample_scores_subject.append(info_from_subject_range)
            true_sample_scores_false_object.append(info_from_false_object_range)
            true_sample_scores_other.append(info_from_subject_range - info_from_false_object_range)
        else:
            false_sample_scores_subject.append(info_from_subject_range)
            false_sample_scores_false_object.append(info_from_false_object_range)
            false_sample_scores_other.append(info_from_subject_range - info_from_false_object_range)

    colors = ['739dd5','f5c780','46C2CB']
    # labels = ['w/o Hallucination','w/ Hallucination']
    labels = ['subject','false object','other']
    # plot_contrasive_scores(true_sample_scores, false_sample_scores,colors,labels,title=)
    # plot_contrasive_scores(
    #     [true_sample_scores, false_sample_scores],
    #     colors,
    #     labels,
    #     title=f"Llama2-{model_size}-chat Information Flow",
    #     save_path=f'{base_dir}/results/information_flow/NobelPrize/tok_pred/llama2-{model_size}-chat_information_flow.pdf'
    # )
    # plot_contrasive_scores(true_sample_scores_other, false_samples_scores_other,colors,labels,title=f"Llama2-{model_size}-chat Information Flow")
    plot_contrasive_scores(
        [false_sample_scores_subject,false_sample_scores_false_object, false_sample_scores_other],
        colors,
        labels,
        title=f"Information Flow w/ Hallucination"
    )
    plot_contrasive_scores(
        [true_sample_scores_subject,true_sample_scores_false_object, true_sample_scores_other],
        colors,
        labels,
        title=f"Information Flow w/o Hallucination"
    )


    # plot_contrasive_scores(true_sample_scores, false_sample_scores)
    # plt.close()
    #
    # plot_scores(true_sample_scores_subject,label="Subject")
    # plot_scores(true_sample_scores_false_object,label="False Object")
    # plt.show()
    # plt.close()
    #
    # plot_scores(false_sample_scores_subject,label="Subject")
    # plot_scores(false_sample_scores_false_object,label="False Object")
    # plt.show()
    # plt.close()
    # print("Debug Usage")
    # pdf_file = f'{result_figs_dir}/information_on_false_object.pdf'
    # print(f"Available Sample Count:{available_count}/{len(samples)}")
    # plot_contrasive_scores(true_sample_scores,false_sample_scores,title=f"Model {model_name} Information Flow On False Object",save_path=pdf_file)
    # print("Saving the result figure file to {}".format(pdf_file))


