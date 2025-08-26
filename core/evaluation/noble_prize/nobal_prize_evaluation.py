from utils import read_json,select
from dataset.ToyDataset.Awards.load_awards import get_id2name,get_sample_id

def cal_acc_who(samples,key='who_question_model_answer',output_key='pred'):
    id2name = get_id2name(samples)
    true_count = 0
    for sample in samples:
        sample_id = get_sample_id(sample)
        groud_truth_names = id2name[sample_id]
        pred = False
        for name in groud_truth_names:
            if name in sample[key]:
                pred = True
                true_count += 1
                break
        sample[output_key] = pred
    acc = true_count / len(samples)
    return acc

def filter(text):
    words = ['I apologize','However, I',"was not awarded",
             "is not awarded","has not been awarded","there is no Nobel",
             "has never been awarded",]
    for word in words:
        if word in text:
            return True
    return False

def cal_acc_who_fp(samples,key='who_question_false_premise_model_answer',output_key='pred'):
    true_count = 0
    for sample in samples:
        flag = filter(text=sample[key])
        sample[output_key] = flag
        if flag:
            true_count += 1
    return true_count / len(samples)

if __name__ == '__main__':
    # Right Answers Evaluation
    model_size = '7b'
    # model_size = '13b'
    result_file =  f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel/llama2-{model_size}-chat_on_nobel_prize_truthful_who_question_model_answer.json'
    results = read_json(result_file)
    print(model_size)
    acc = cal_acc_who(results,key='who_question_model_answer')
    print("who acc:",acc)

    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/nobel/llama2-{model_size}-chat_on_nobel_prize_truthful_who_question_false_premise_model_answer.json'
    results = read_json(result_file)
    fp_acc = cal_acc_who_fp(results,key='who_question_false_premise_model_answer')
    print('who fp acc',fp_acc)

