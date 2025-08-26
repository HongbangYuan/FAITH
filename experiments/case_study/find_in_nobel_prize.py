from experiments.case_study.find_in_film_release import collect


if __name__ == '__main__':
    from core.evaluation.noble_prize.nobal_prize_evaluation import cal_acc_who,cal_acc_who_fp
    from core.evaluation.noble_prize.nobel_prize_when_evaluation import cal_when_acc
    from core.evaluation.noble_prize.nobel_prie_when_fp_evaluation import cal_when_fp_acc
    from experiments.case_study.find_in_film_release import collect
    from utils import read_json

    # model_size = '7b'
    model_size = '13b'
    print("model_size:",model_size)

    pp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/attention/iti/nobel_prize/llama2-{model_size}-chat_on_when_question.json'
    pp_samples = read_json(pp_file)

    fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/attention/iti/nobel_prize/llama2-{model_size}-chat_on_when_fp_question.json'
    fp_samples = read_json(fp_file)

    when_acc = cal_when_acc(pp_samples, key='when_question_model_answer')
    when_fp_acc = cal_when_fp_acc(fp_samples, key='when_fp_question_model_answer')
    print("when_acc",when_acc)
    print("when_fp_acc",when_fp_acc)

    results = collect(pp_samples,fp_samples)
    for result in results:
        print(len(result))

    # results = collect(pp_samples, fp_samples)
    # not_nomination_count = [sample for sample in results[1] if 'nominat' in sample['why_question_false_premise_model_answer']]
    # not_exist_count = [sample for sample in results[1] if sample not in not_nomination_count and 'did not exist' in sample['why_question_false_premise_model_answer']]
    # a = [sample for sample in results[1] if sample not in not_nomination_count and sample not in not_exist_count]
    # answers = [s['when_fp_question_model_answer'] for s in results[2]]
    answers = [(s['when_fp_question_model_answer'],s['when_question_model_answer']) for s in results[0]]
