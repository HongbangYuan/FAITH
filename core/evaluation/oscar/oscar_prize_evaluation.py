

def cal_acc_yes_no(samples,key,output_key='pred'):
    true_count = 0
    for sample in samples:
        answer = sample[key]
        if 'yes' in answer.lower():
            pred = True
            true_count += 1
        else:
            pred = False
        sample[output_key] = pred
    return true_count/len(samples)

def cal_acc_who(samples,key,output_key='pred'):
    true_count = 0
    for sample in samples:
        answer = sample[key]
        if sample["name"].lower() in answer.lower():
            pred = True
            true_count += 1
        else:
            pred = False
        sample[output_key] = pred
    return true_count/len(samples)

def cal_acc_why_fp(samples,key,output_key='pred'):
    words = ['I apologize', "However, I must point out","However, I think there may be some inaccuracies",'However, I notice that the question itself may not be',
             'However, I noticed that the question contains some inaccuracies',"However, I would like to point out","However, I notice that the question",
             "However, I noticed that the question","However"]

    def filter(text):
        for word in words:
            if word in text:
                return True
        return False

    true_count = 0
    for sample in samples:
        answer = sample[key]
        if filter(answer):
            true_count+=1
            pred = True
        else:
            pred = False
        sample[output_key] = pred
    return true_count/len(samples)


if __name__ == '__main__':
    from utils import read_json

    model_size = '7b'
    print("model_size:",model_size)
    # result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/oscar'
    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/oscar'
    mode = 'untruthful'
    print(model_size,mode)
    # yes_no_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_{mode}_yes_no_question_model_answer.json'
    # yes_no_wrong_year_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_{mode}_yes_no_question_wrong_year_model_answer.json'
    # who_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_{mode}_who_question_model_answer.json'
    # which_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_{mode}_which_question_model_answer.json'
    # why_false_premise_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_{mode}_why_question_false_premise_model_answer.json'
    why_long_false_premise_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_why_long_question_false_premise_model_answer.json'

    # # yes_no_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_yes_no_question_model_answer.json'
    # samples = read_json(yes_no_file)
    # yes_no_acc = cal_acc_yes_no(samples,key='yes_no_question_model_answer')
    # print("yes_no_acc",yes_no_acc)
    #
    # # yes_no_wrong_year_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_yes_no_question_wrong_year_model_answer.json'
    # samples = read_json(yes_no_wrong_year_file)
    # yes_no_wrong_year_acc = 1 - cal_acc_yes_no(samples,key='yes_no_question_wrong_year_model_answer')
    # print("yes_no_wrong_year_acc",yes_no_wrong_year_acc)
    #
    # # who_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_who_question_model_answer.json'
    # samples = read_json(who_file)
    # who_acc = cal_acc_who(samples,key='who_question_model_answer')
    # print("who_acc",who_acc)
    #
    # # which_file = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_which_question_model_answer.json'
    # samples = read_json(which_file)
    # which_acc = cal_acc_who(samples,key='which_question_model_answer')
    # print("which_acc",which_acc)


    # why_false_premise = f'{result_path}/llama2-{model_size}-chat_on_oscar_prize_why_question_false_premise_model_answer.json'
    samples = read_json(why_long_false_premise_file)

    why_fp_acc = cal_acc_why_fp(samples,key="why_long_question_false_premise_model_answer")
    print("why_fp_acc",why_fp_acc)
    a = [sample["why_long_question_false_premise_model_answer"] for sample in samples]
