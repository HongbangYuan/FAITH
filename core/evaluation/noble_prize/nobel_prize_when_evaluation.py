
def judge_single_sample_when_question(sample,pred_key,ground_truth_key):
    pred = sample[pred_key]
    ground_truth = str(sample[ground_truth_key])
    if ground_truth in pred:
        return True
    else:
        return False


def cal_when_acc(samples,key,ground_truth_key='when_answer_ground_truth',result_key='pred'):
    true_count = 0
    num_samples = len(samples)
    for sample in samples:
        judge = judge_single_sample_when_question(sample,key,ground_truth_key)
        if judge:
            true_count += 1
        sample[result_key] = judge
    return true_count / num_samples


if __name__ == '__main__':
    from utils import read_json

    model_size = '7b'
    # model_size = '13b'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/dola/nobel_prize/llama2-13b-chat_on_nobel_prize_when_question'
    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/nobel_prize/llama2-{model_size}-chat_on_nobel_prize_when_question_model_answer.json'
    print("model_size",model_size)

    samples = read_json(result_file)

    pred_key = 'when_question_model_answer'
    ground_truth_key = 'when_answer_ground_truth'
    result_key = "when_answer_eval"
    acc = cal_when_acc(samples,pred_key,ground_truth_key,result_key)

    print("acc:",acc)
    answers = [s[pred_key] for s in samples if s[result_key] == True]

