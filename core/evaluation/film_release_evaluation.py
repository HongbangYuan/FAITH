from utils import select
import re


def find_years_in_strings(strings):
    year_pattern = r'\b\d{4}\b'  # This regex pattern matches 4-digit numbers (assumed to be years)

    result = []
    for s in strings:
        matches = re.findall(year_pattern, s)
        result.append(list(set(matches)))

    return result


def cal_acc(samples, key='false_premise_model_answer', output_key='pred'):
    answers = [sample[key] for sample in select(samples)]
    true_count = 0
    all_count = len(samples)
    years = find_years_in_strings(answers)
    for sample, year in zip(samples, years):
        if sample["release_year"] in year:
            true_count += 1
            sample["outptu_key"] = True
        else:
            sample["outptu_key"] = False
    return true_count / all_count


def cal_acc_wiki_movies(samples, key, output_key='pred'):
    answers = [sample[key] for sample in samples]
    true_count = 0
    all_count = len(samples)
    years = find_years_in_strings(answers)
    for sample, year in zip(samples, years):
        flag = cal_acc_wiki_movie_single_sample(sample, key, output_key)
        if flag:
            true_count += 1
    return true_count / all_count


def cal_acc_wiki_movie_single_sample(sample, key, output_key='pred'):
    flag = False
    for time in sample["time"]:
        if str(time) in sample[key]:
            flag = True
    sample[output_key] = flag
    return flag


def judge_per_answer(answer, ground_truth):
    flag = False
    for time in ground_truth:
        if str(time) in answer:
            flag = True
    return flag


if __name__ == '__main__':
    from utils import read_json, select, write_to_json, merge_multiple_json
    print("Start Running!")

    # model_size = '13b'
    # print(f"model_size:{model_size}")
    # topk = 15 if model_size == '13b' else 5
    # for from_q in [1,2,3,4]:
    #     for curr_q in [1,2,3,4]:
    #         result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout/Movies/more_fps/across_knowledge/llama2-{model_size}-chat_on_wiki_movie_Q_{curr_q}_with_head_from_prize_Q{from_q}_threshold_0.1_topk_{topk}.json'
    #         samples = read_json(result_file)
    #         # print("Debug Usage")
    #         input_key = f'knock_out_question_{curr_q}'
    #         acc = cal_acc_wiki_movies(samples,input_key)
    #         print(f"from_prize_q{from_q} curr_q{curr_q}    acc={acc:.4f}")
    #     print("-----------------")


    # model_size = '7b'
    # print("model_size")
    # topk = 15 if model_size == '13b' else 5
    # for from_q in [1,2,3,4]:
    #     for curr_q in [1,2,3,4]:
    #         # if curr_q != from_q:
    #         if True:
    #             result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout/Movies/more_fps/within_knowledge/llama2-{model_size}-chat_on_wiki_movie_Q_{curr_q}_with_head_from_Q{from_q}_threshold_0.1_topk_{topk}.json'
    #             samples = read_json(result_file)
    #             # print("Debug Usage")
    #             input_key = f'knock_out_question_{curr_q}'
    #             acc = cal_acc_wiki_movies(samples,input_key)
    #             print(f"from_q{from_q} curr_q{curr_q}    acc={acc:.4f}")
    #     print("-----------------")


    # model_size = '7b'
    # print(model_size)
    for question_num in [1,2,3,4]:
        question_num = str(question_num)
        # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout/Movies/more_fps/llama2-13b-chat_on_wiki_movie_knock_out_heads_{question_num}.json'
        # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/repe/Movies/{model_size}/llama2-{model_size}-chat_on_wiki_movie_fp_question_{question_num}_repe_answer.json'
        # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/iti/Movies/{model_size}/llama2-{model_size}-chat_on_nobel_prize_iti_fp_question_{question_num}_iti_answer.json'
        result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/Movies/more_fps/all_pos/llama2-7b-chat_on_wiki_movie_knock_out_random_heads_{question_num}_all_pos.json'
        # result_file = f'/netdisk/hongbang/projects/HalluInducing/results/causal_trace/knockout/Movies/more_fps/all_pos/llama2-{model_size}-chat_on_wiki_movie_knock_out_heads_{question_num}_top_15_all_pos.json'
        samples = read_json(result_file)
        # input_key = f'fp_question_{question_num}_repe_answer'
        # input_key = f'fp_question_{question_num}_iti_answer'
        input_key = f'knock_out_question_{question_num}'
        acc = cal_acc_wiki_movies(samples,key=input_key)
        print(f"acc={acc:.4f}")

        # 挑选出之前做错后来作对的样本
        before_samples = read_json("/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-7b-chat_on_wiki_movies_more_fp_fp_question_4_model_answer.json")
        selected = []
        for idx,sample in enumerate(samples):
            original_key = f"fp_question_{question_num}_model_answer"
            original_flag = cal_acc_wiki_movie_single_sample(before_samples[idx],key=original_key)
            curr_flag = cal_acc_wiki_movie_single_sample(sample,key=f"knock_out_question_{question_num}")
            if original_flag == 0 and curr_flag == 1:
                sample["idx"] = idx
                sample["original_answer"] = before_samples[idx][original_key]
                selected.append(sample)

    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/dola/Movies/llama2-13b-chat_on_wiki_movie_dola_fp_question_4'
    # samples = read_json(result_file)
    #
    # for num in [1,2,3,4]:
    #     key = f'fp_question_{num}_dola_answer'
    #     print("question num:",num)
    #     acc = cal_acc_wiki_movies(samples, key=key)
    #     print(f"Acc:{acc:.4f}", )

    # model_size = '7b'
    # model_size = '13b'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{model_size}-chat_on_film_release_dataset_untruthful_instruction.json'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/iti/film_release/llama2-{model_size}-chat-iti_on_film_release_when_question.json'
    # print(model_size)
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/repe/film_release/llama2-{model_size}-chat_on_film_release_when_question_model_answer.json'
    # result_file1 = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    # result_file1 = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    # result_file2 = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_1001_to_5494_why_fp_question_model_answer.json'
    # tok_result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/tok_pred/llama2-{model_size}-chat_on_wiki_movies_0_to_1000_why_fp_question_token_answer.json"
    # samples = read_json(tok_result_file)
    #
    # tok_result_key = 'token_pred'
    #
    # token_predictions = [sample[tok_result_key][0] for sample in samples]
    # token_acc = sum(token_predictions) / len(samples)
    # print("token_acc:",token_acc)

    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/knockout/llama2-{model_size}-chat_on_movies_knock_out.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/dola/movies/llama2-7b-chat_on_wiki_movie_dola_why_fp_question'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_why_fp_question_model_answer.json'
    # merge_multiple_json([result_file1,result_file2],result_file)
    # samples = read_json(result_file)

    # samples_one_time = [sample for sample in orig_samples if len(sample["time"]) == 1]
    # samples = [sample for sample in samples_one_time if
    #                     str(sample["time"][0])[:-1] == str(sample["false_year"])[:-1]]

    # key = 'why_fp_question_model_answer_dola'
    # key = 'knockout_result'
    # print("Acc:", cal_acc_wiki_movies(samples, key=key))
    # print("original acc:", cal_acc_wiki_movies(samples, key='why_fp_question_model_answer'))
    # print("After knock out acc:",cal_acc_wiki_movies(samples,key=key))
    # key = 'why_fp_question_model_answer'
    #
    # acc = cal_acc_wiki_movies(samples, key=key,output_key='fp_pred')
    # # acc = cal_acc(samples,key='false_premise_question_model_answer')
    #
    # print("Acc:{}".format(acc))
    # false_samples = select(samples,fp_pred=False)
    # true_samples = select(samples,fp_pred=True)
    # answers = [sample[key] for sample in false_samples]
    # true_answers = [sample[key] for sample in true_samples]
    # analyze_samples = [sample if sample["fp_pred"] == False else None for sample in samples]
    # # write_to_json(samples,f'/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{model_size}-chat_on_film_release_dataset_evaluation.json')
