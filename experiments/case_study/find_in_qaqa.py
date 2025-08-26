

if __name__ == '__main__':
    from core.evaluation.qaqa_finegrained import fine_grained_evaluation,print_table,calc_acc,cal_acc_yes_no
    from utils import read_json
    from utils import select
    from experiments.case_study.find_in_film_release import collect

    print("Hello World!")

    model_size = '13b'
    # model_size = '7b'
    print(model_size)

    pp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-{model_size}-chat_yesno_verification_question_including_valid_qs_new_format.json'
    fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-{model_size}-chat_new_format_evaluation.json'

    fp_samples = select(read_json(fp_file),all_assumptions_valid='has_invalid')
    for sample in fp_samples:
        sample["fp_model_answer"] = sample["model_answer"]

    acc = calc_acc(fp_samples)

    pp_samples = select(read_json(pp_file),all_assumptions_valid='has_invalid')
    acc_yes_no = cal_acc_yes_no(pp_samples,key='model_answer')
    for sample in pp_samples:
        sample["pp_model_answer"] = sample["model_answer"]
    print("acc",acc)
    print("acc_yes_no",acc_yes_no)

    results = collect(pp_samples,fp_samples,fp_pred_key='relaxed_result')
    print(" ".join([str(len(result)) for result in results]))

