from experiments.case_study.find_in_film_release import collect


if __name__ == '__main__':
    from core.evaluation.oscar.oscar_prize_evaluation import cal_acc_who,cal_acc_why_fp
    from utils import read_json

    pp_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/oscar/llama2-13b-chat_on_oscar_prize_who_question_model_answer.json'
    pp_samples = read_json(pp_file)

    fp_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/oscar/llama2-13b-chat_on_oscar_prize_why_long_question_false_premise_model_answer.json'
    fp_samples = read_json(fp_file)

    cal_acc_who(pp_samples, key='who_question_model_answer')
    cal_acc_why_fp(fp_samples, key='why_long_question_false_premise_model_answer')

    results = collect(pp_samples, fp_samples)
    # not_nomination_count = [sample for sample in results[1] if 'nominat' in sample['why_question_false_premise_model_answer']]
    # not_exist_count = [sample for sample in results[1] if sample not in not_nomination_count and 'did not exist' in sample['why_question_false_premise_model_answer']]
    # a = [sample for sample in results[1] if sample not in not_nomination_count and sample not in not_exist_count]
    answers = [s['why_long_question_false_premise_model_answer'] for s in results[1]]
    # answers = [s['who_question_model_answer'] for s in results[1]]
