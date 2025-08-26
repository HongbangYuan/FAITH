def cal_when_fp_acc(samples, key, result_key='pred'):
    true_count = 0
    for sample in samples:
        answer = sample[key]
        ground_truth_year = sample["when_answer_ground_truth"]
        if str(ground_truth_year) in answer:
            flag = True
            true_count += 1
        else:
            flag = False
        sample[result_key] = flag
    return true_count / len(samples)


def cal_when_fp_per_sample(sample, answer_key, ground_truth_key='when_answer_ground_truth'):
    answer = sample[answer_key]
    ground_truth_year = sample[ground_truth_key]
    flag = judge_per_answer(answer,ground_truth_year)
    return flag

def judge_per_answer(answer,ground_truth):
    if str(ground_truth) in answer:
        flag = True
    else:
        flag = False
    return flag


if __name__ == '__main__':
    from utils import read_json
    from dataset.ToyDataset.Awards.load_awards import load_nobel_prize_only_fp

    # model_size = '7'
    # model_size = '13'
    # print("model_size",model_size)
    # # # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/dola/NobelPrize/llama2-{model_size}b-chat_on_wiki_movie_dola_when_fp_question4'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/repe/NobelPrize/{model_size}b/llama2-{model_size}b-chat_on_nobel_prize_when_fp_question4_repe_answer.json'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/iti/NobelPrize/{model_size}b/llama2-{model_size}b-chat_on_nobel_prize_iti_when_fp_question4_iti_answer.json'
    # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/iti/NobelPrize/7b/update_trainset/llama2-7b-chat_on_nobel_prize_iti_when_fp_question4_iti_answer.json'
    # samples = read_json(result_file)
    # #
    # for num in [1,2,3,4]:
    #     if num == 1:
    #         num = ""
    #     key = f'when_fp_question{num}_iti_answer'
    #     # key = f'when_fp_question{num}_repe_answer'
    #     print("question num:",num)
    #     acc = cal_when_fp_acc(samples, key=key,)
    #     print(f"Acc:{acc:.4f}", )
    #
    #
    # acc = cal_when_fp_acc(samples,key="when_fp_question2_repe_answer",result_key='pred')
    # answers = [sample["when_fp_question2_repe_answer"] for sample in samples if sample['pred']]
    # analyze_samples = [(sample["awardYear"],sample["when_fp_question2_repe_answer"]) for sample in samples if sample['pred']]


    # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/all_pos/llama2-13b-chat_on_nobel_prize_knock_out_heads_3_more_pos.json'
    # # result_file = '/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/llama2-7b-chat_on_nobel_prize_knock_out_heads_3.json'
    # samples = read_json(result_file)
    # question_num = '3'
    # input_key = f'knock_out_question_{question_num}'
    # acc = cal_when_fp_acc(samples, key=input_key, result_key='pred')
    # print(f"acc={acc:.6f}")

    # # question_num = '3'
    for question_num in [1,2,3,4]:
        question_num = str(question_num)
        # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/all_pos/llama2-13b-chat_on_nobel_prize_knock_out_heads_{question_num}_fix_pos_knock_all_pos_top10_threshold_0.05.json'
        # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/all_pos/llama2-13b-chat_on_nobel_prize_knock_out_heads_{question_num}_fix_pos_knock_all_pos_top15.json'
        # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/all_pos/llama2-13b-chat_on_nobel_prize_knock_out_heads_{question_num}_fix_pos_knock_all_pos_top20_threshold0.1.json'
        result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/all_pos/llama2-7b-chat_on_nobel_prize_knock_out_heads_{question_num}_fix_pos_knock_all_pos_top5_threshold0.1.json'
        # result_file = '/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/NobelPrize/more_fps/llama2-7b-chat_on_nobel_prize_knock_out_heads_3.json'
        samples = read_json(result_file)
        input_key = f'knock_out_question_{question_num}'
        acc = cal_when_fp_acc(samples,key=input_key,result_key='pred')
        print(f"acc={acc:.4f}")

        # 挑选出之前做错后来作对的样本
        before_samples = read_json("/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/nobel_prize/llama2-13b-chat_on_nobel_prize_more_fp_when_fp_question4_model_answer.json")
        curr_ids = set([sample["motivation"] for sample in samples])
        before_samples = [sample for sample in before_samples if sample["motivation"] in curr_ids]
        selected = []
        for idx,sample in enumerate(samples):
            if question_num == '1':
                question_num = ''
            original_sample = before_samples[idx]
            # original_key = f"when_fp_question{question_num}_model_answer"
            # original_flag = cal_when_fp_per_sample(before_samples[idx],answer_key=original_key)
            curr_flag = cal_when_fp_per_sample(sample,answer_key=input_key)
            if  curr_flag == 1:
                sample["idx"] = idx
                # sample["original_answer"] = before_samples[idx][original_key]
                selected.append(sample)


    # for k in [2,3,4]:
    #     # print(f"question k={k}")
    #     acc = cal_when_fp_acc(samples, key=f"when_fp_question{k}_model_answer", result_key=f'pred_{k}')
    #     print(f"acc for when fp question {k}={acc:.6f}")

    # model_size = '7b'
    # model_size = '13b'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout_generation/NobelPrize/llama2-{model_size}-chat_on_nobel_prize_knock_out_different_heads_45231.json'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout_generation/NobelPrize/llama2-{model_size}-chat_on_nobel_prize_knock_out_different_heads_43521.json'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout_generation/NobelPrize/llama2-7b-chat_on_nobel_prize_knock_out_heads_54312.json'
    # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout_generation/NobelPrize/llama2-7b-chat_on_nobel_prize_knock_out_heads.json'
    # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout_generation/NobelPrize/llama2-7b-chat_on_nobel_prize_knock_out_heads.json'
    # samples = read_json(result_file)
    # for k in range(1,5+1):
    # # for k in [2,3]:
    #     input_key = f'knockout_result_{k}'
    #     acc = cal_when_fp_acc(samples, key=input_key, result_key=f'pred_{k}')
    #     print(f"acc_{k}:{acc}")
    #     answers = [sample[input_key] for sample in samples]
    # selected_samples = [sample for sample in samples if sample["pred_5"] and not sample["pred_1"]]
    # key = 'knockout_result'
    # acc = cal_when_fp_acc(samples, key=key)
    # print(f"acc:{acc}")
    # answers = [sample[key] for sample in samples if sample["pred"]]

    # another_acc = sum([cal_when_fp_per_sample(sample,input_key) for sample in samples])/len(samples)
    # assert another_acc == acc

    # # model_size = '7b'
    # model_size = '13b'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/nobel_prize/llama2-{model_size}-chat_on_nobel_prize_when_fp_question_model_answer.json'
    # samples_7b = load_nobel_prize_only_fp(model_size=7)
    # samples_13b = load_nobel_prize_only_fp(model_size=13)
    # print("model_size",model_size)
    #
    # samples = read_json(result_file)
    #
    # input_key = 'when_fp_question_model_answer'
    # acc = cal_when_fp_acc(samples_13b,key=input_key,result_key='pred')
    #
    # print("acc:",acc)
    # # a = [s[input_key] for s in samples if s['pred']==False]
