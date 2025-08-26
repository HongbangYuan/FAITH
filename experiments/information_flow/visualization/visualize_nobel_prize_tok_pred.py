

if __name__ == '__main__':
    import torch
    from torch.nn import CrossEntropyLoss
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, select, get_model_name_mapping, load_llama_tokenizer,read_pickle,write_to_pickle
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from experiments.information_flow.cal_saliency_score_nobel_prize import reshape_saliency_score ,remove_prompt \
        ,format_question_answer ,untuple
    from experiments.causal_trace.path_patch.single_tok_pred_nobel_prize import format_answer_from_sample
    from experiments.information_flow.information_from_author import find_token_range,plot_contrasive_scores,plot_scores
    from functools import partial
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp
    import argparse
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/home/zhuoran/hongbang/TimesNewRoman/TIMES.TTF')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

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
    tokenizer = load_llama_tokenizer(model_name_or_path)


    results = []
    true_sample_scores_subject = []
    true_sample_scores_false_object = []
    false_sample_scores_subject = []
    false_sample_scores_false_object = []
    true_sample_scores = []
    false_sample_scores = []
    true_sample_scores_other = []
    false_sample_scores_other = []
    available_count = 0
    for idx, sample in enumerate(tqdm(samples)):
        sample_name = sample[subject_key].replace('/' ,'').replace(" " ,'_')
        filename = f"{result_dir}/{idx}_{sample_name}.npz"
        if not os.path.isfile(filename):
            continue
        try:
            result = dict(np.load(filename,allow_pickle=True))
        except:
            continue
        saliency_score = result["saliency_score"]
        available_count += 1

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
        orig_prompt = format_question_answer(question, uncompleted_answer)
        generation = tokenizer(orig_prompt, return_tensors="pt")["input_ids"][0]
        # _,end_of_question = find_token_range(tokenizer,generation,"[/INST]")

        subject_range = find_token_range(tokenizer,generation,sample[subject_key])
        false_object_range = find_token_range(tokenizer,generation,str(false_year))
        _, end_of_question = find_token_range(tokenizer, generation, "[/INST]")

        other_range = torch.ones(saliency_score.shape[-1],dtype=torch.bool)
        other_range[subject_range[0]:subject_range[-1]] = False
        other_range[false_object_range[0]:false_object_range[-1]] = False
        # info_from_other = saliency_score[:,end_of_question:,other_range].mean(axis=(-1,-2))
        # info_from_subject_range = saliency_score[:,end_of_question:,subject_range[0]:subject_range[-1]].mean(axis=(-1,-2))
        # info_from_false_object_range = saliency_score[:,end_of_question:,false_object_range[0]:false_object_range[-1]].mean(axis=(-1,-2))

        info_from_other = saliency_score[:,-1,other_range].mean(axis=-1)
        info_from_subject_range = saliency_score[:,-1,subject_range[0]:subject_range[-1]].mean(axis=-1)
        info_from_false_object_range = saliency_score[:,-1,false_object_range[0]:false_object_range[-1]].mean(axis=-1)



        if sample[fp_result_key]:
            true_sample_scores_subject.append(info_from_subject_range)
            true_sample_scores_false_object.append(info_from_false_object_range)
            true_sample_scores.append(info_from_subject_range - info_from_false_object_range)
            true_sample_scores_other.append(info_from_other)
        else:
            false_sample_scores_subject.append(info_from_subject_range)
            false_sample_scores_false_object.append(info_from_false_object_range)
            false_sample_scores.append(info_from_subject_range - info_from_false_object_range)
            false_sample_scores_other.append(info_from_other)

    # result_file = f'{base_dir}/results/information_flow/NobelPrize/tok_pred/infor_flow_{model_size}.pkl'
    # write_to_pickle([true_sample_scores,false_sample_scores],result_file)

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
    if model_size == '7b':
        limit = [0,20]
    else:
        limit = [0,10]
    plot_contrasive_scores(
        [false_sample_scores_subject,false_sample_scores_false_object, false_sample_scores_other],
        colors,
        labels,
        title=f"Llama-2-{model_size}-chat w/ Hallucination",
        ylimit=limit
    )
    plot_contrasive_scores(
        [true_sample_scores_subject,true_sample_scores_false_object, true_sample_scores_other],
        colors,
        labels,
        title=f"Llama-2-{model_size}-chat w/o Hallucination",
        ylimit=limit
    )
    write_to_pickle(
        [true_sample_scores_subject,true_sample_scores_false_object, true_sample_scores_other,false_sample_scores_subject,false_sample_scores_false_object, false_sample_scores_other],
        f'/home/zhuoran/hongbang/visualize_nobel_prize_tok_pred_info_flow_{model_size}.pkl',
    )

    #
    # title = f"Llama2-{model_size}-chat Information Flow"
    # # scores = [false_sample_scores_subject,false_sample_scores_false_object, false_samples_scores_other]
    # scores = [false_sample_scores_subject,false_sample_scores_false_object, false_sample_scores_other]
    # # scores = [true_sample_scores_subject,true_sample_scores_false_object, true_sample_scores_other]
    # fig, ax = plt.subplots()
    # for i,(color,label) in enumerate(zip(colors,labels)):
    #     plot_scores(scores[i], label=labels[i],color='#'+colors[i])
    # plt.xlabel('Layers',fontsize=16)
    # plt.ylabel('Information Flow',fontsize=16)
    # plt.axvline(x=15, color='#F2F7A1', linestyle='--')
    # plt.axvline(x=25, color='#F2F7A1', linestyle='--')
    # plt.legend(fontsize=16)
    # if title:
    #     plt.title(title,fontsize=20)
    # # if save_path:
    # #     plt.savefig(save_path)
    # #     plt.close()
    # # else:
    # plt.show()

    # print("Debug Usage")
    #
    # colors =
    # fig, ax = plt.subplots()
    # plt.title(f"Llama2-{model_size}-chat Information Flow",fontsize=20)
    #
    # label = 'w/o Hallucination'
    # color_0 = '#'+colors[0]
    # data = np.stack(true_sample_scores)  # (num_samples,num_layers)
    # # Use seaborn's lineplot with the mean and confidence interval
    # mean_values = np.mean(data, axis=0)
    # ci_values = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    # # Create x-axis values
    # x_values = np.arange(1, data.shape[-1] + 1)
    # # Plot the mean line
    # plt.plot(x_values, mean_values, label=label,color=color_0)
    #
    # # Fill between the upper and lower bounds of the confidence interval
    # # plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2, label='95% CI')
    # plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2,color=color_0)
    # # plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values)
    #
    # # Set plot labels and title
    # plt.xlabel('Layers',fontsize=16)
    # plt.ylabel('Information Flow',fontsize=16)
    # # plt.title('Line Plot with Mean and 95% Confidence Interval')
    #
    # # Show legend
    # plt.legend()
    #
    # label = 'w/ Hallucination'
    # color_1 = '#'+colors[1]
    # data = np.stack(false_sample_scores)  # (num_samples,num_layers)
    # # Use seaborn's lineplot with the mean and confidence interval
    # mean_values = np.mean(data, axis=0)
    # ci_values = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    # # Create x-axis values
    # x_values = np.arange(1, data.shape[-1] + 1)
    # # Plot the mean line
    # plt.plot(x_values, mean_values, label=label,color='#'+colors[1])
    #
    # # Fill between the upper and lower bounds of the confidence interval
    # # plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2, label='95% CI')
    # plt.fill_between(x_values, mean_values - ci_values, mean_values + ci_values, alpha=0.2,color=color_1)
    #
    #
    # ax.tick_params(axis='both', which='major', labelsize=13)
    # # Show legend
    # plt.legend(fontsize=16)
    #
    # plt.show()
    # plt.savefig(f'{base_dir}/results/information_flow/NobelPrize/tok_pred/llama2-{model_size}-chat_information_flow.pdf')


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
