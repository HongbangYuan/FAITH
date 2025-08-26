



if __name__ == '__main__':
    from utils import read_json,write_to_json
    from core.evaluation.noble_prize.nobel_prize_when_evaluation import cal_when_acc
    from core.evaluation.noble_prize.nobel_prie_when_fp_evaluation import cal_when_fp_acc

    model_size = '7b'
    # model_size = '13b'
    # result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/dola/nobel_prize/llama2-13b-chat_on_nobel_prize_when_question'
    when_result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/nobel_prize/llama2-{model_size}-chat_on_nobel_prize_when_question_model_answer.json'
    when_fp_result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/nobel_prize/llama2-{model_size}-chat_on_nobel_prize_when_fp_question_model_answer.json'
    print("model_size",model_size)

    samples_when = read_json(when_result_file)
    pred_key = 'when_question_model_answer'
    ground_truth_key = 'when_answer_ground_truth'
    when_result_key = "when_answer_eval"
    acc = cal_when_acc(samples_when,pred_key,ground_truth_key,when_result_key)
    print("acc:",acc)
    answers = [s[pred_key] for s in samples_when if s[when_result_key] == True]

    samples_when_fp = read_json(when_fp_result_file)
    input_key = 'when_fp_question_model_answer'
    when_fp_result_key = 'when_fp_answer_eval'
    acc = cal_when_fp_acc(samples_when_fp,key=input_key,result_key=when_fp_result_key)
    print("acc:",acc)
    a = [s[input_key] for s in samples_when_fp if s[when_fp_result_key]==False]

    samples = []
    count = 0
    for sample1,sample2 in zip(samples_when,samples_when_fp):
        if sample1[when_result_key]:
            if not sample2[when_fp_result_key]:
                count += 1
            samples.append({
                **sample1,**sample2
            })

    print(f"Create a total of {len(samples)} samples.")
    print(f"Wrong FP samples:{count}")
    selected_samples = [sample for sample in samples if sample[when_fp_result_key] == True]
    # file = f"/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Awards/llama2-{model_size}-chat_on_nobel_prize.json"
    # write_to_json(samples,file)
    # print("Writing created dataset to {}".format(file))

    answers = [x['when_fp_question_model_answer'] for x in selected_samples]