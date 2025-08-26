
def judge_single_sample_where_question(sample,pred_key,ground_truth_key):
    pred = sample[pred_key]
    ground_truths = sample[ground_truth_key]
    flags = []
    for ground_truth in ground_truths:
        # nan in float caused this problem
        if isinstance(ground_truth,float):
            continue
        if ground_truth in pred:
            flags.append(True)
        else:
            flags.append(False)
    if False not in flags:
        return True
    else:
        return False


def cal_where_acc(samples,pred_key,ground_truth_key,result_key):
    true_count = 0
    num_samples = len(samples)
    for i,sample in enumerate(samples):
        # print(i)
        judge = judge_single_sample_where_question(sample,pred_key,ground_truth_key)
        if judge:
            true_count += 1
        sample[result_key] = judge
    return true_count / num_samples


if __name__ == '__main__':
    from utils import read_json

    # model_size = '7b'
    model_size = '13b'
    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel/llama2-{model_size}-chat_on_nobel_prize_untruthful_where_question_actually_model_answer.json'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/llama2-{model_size}-chat_on_film_noble_prize_where_question.json'
    print("model_size",model_size)

    samples = read_json(result_file)


    pred_key = 'where_question_actually_model_answer'
    # pred_key = 'where_answer'
    ground_truth_key = 'where_answer_ground_truth'
    result_key = "where_answer_eval"
    acc = cal_where_acc(samples,pred_key,ground_truth_key,result_key)

    print("acc:",acc)
    answers = [s[pred_key] for s in samples]
