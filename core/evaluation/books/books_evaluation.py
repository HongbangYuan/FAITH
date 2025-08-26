import random


def check_who(ground_truths,answer):
    for ground_truth in ground_truths:
        if ground_truth.lower() in answer.lower():
            return True

    return False

def cal_acc_who(samples,key='who_question_model_answer',output_key='pred'):
    true_count = 0
    for sample in samples:
        flag = check_who(sample["who_question_ground_truth"],sample[key])
        sample[output_key] = flag
        if flag:
            true_count += 1
    return true_count/len(samples)

def filter(text):
    pass

def cal_acc_when_fp(samples,key="when_false_premise_question_model_answer",output_key='pred'):
    true_count = 0
    for sample in samples:
        flag = check_who(sample["who_question_ground_truth"],sample[key])
        sample[output_key] = flag
        if flag:
            true_count += 1
    return true_count/len(samples)

if __name__ == '__main__':
    from utils import read_json,write_to_json
    from experiments.case_study.find_in_film_release import collect
    from random import shuffle
    import random
    random.seed(0)

    model_size = '13b'
    # model_size = '7b'
    print(model_size)

    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results'
    file = f'{base_dir}/causal_trace/knockout/Books/more_fps/llama2-13b-chat_on_books_knock_out_heads.json'
    samples = read_json(file)
    key = 'knock_out_question2'
    acc = cal_acc_when_fp(samples,key)
    print("acc:",acc)

    # pp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_who_question_model_answer.json'
    # pp_samples = read_json(pp_file)
    # acc_who = cal_acc_who(pp_samples)
    # print("acc_who",acc_who)
    #
    # fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-{model_size}-chat_on_books_author_when_false_premise_question_model_answer.json'
    # fp_samples = read_json(fp_file)
    # answers = [sample["when_false_premise_question_model_answer"] for sample in fp_samples]
    # acc_when_fp = cal_acc_when_fp(fp_samples)
    # print("acc_when_fp",acc_when_fp)
    #
    # results = collect(pp_samples,fp_samples)
    # print(" ".join([str(len(result)) for result in results]))
    #
    # idx = 3
    # answers_tp = [s["who_question_model_answer"] for s in results[idx]]
    # answers_fp = [s["when_false_premise_question_model_answer"] for s in results[idx]]
    # answers_compare = [(a1,a2) for a1,a2 in zip(answers_tp,answers_fp)]

    # # write type1 and type4 to fil
    # samples = results[0] + results[3]
    # for sample in samples:
    #     sample["tp_pred"] = sample["pp_pred"]
    #     del sample["pp_pred"]
    # shuffle(samples)
    # write_to_json(samples,f"/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Books/llama2-{model_size}-chat_books.json")
