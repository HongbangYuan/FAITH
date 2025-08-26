
def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def collect(pp_samples, fp_samples, pp_pred_key='pred', fp_pred_key='pred'):
    pp_true_fp_true = []
    pp_true_fp_false = []
    pp_false_fp_true = []
    pp_false_fp_false = []
    for pp_sample, fp_sample in zip(pp_samples, fp_samples):
        pp_pred = pp_sample[pp_pred_key]
        fp_pred = fp_sample[fp_pred_key]
        sample = without_keys({**pp_sample, **fp_sample, "pp_pred": pp_pred, "fp_pred": fp_pred},
                              [pp_pred_key, fp_pred_key])
        if pp_pred and fp_pred:
            pp_true_fp_true.append(sample)
        elif pp_pred and not fp_pred:
            pp_true_fp_false.append(sample)
        elif not pp_pred and fp_pred:
            pp_false_fp_true.append(sample)
        else:
            pp_false_fp_false.append(sample)
    return pp_true_fp_true, pp_false_fp_true, pp_false_fp_false, pp_true_fp_false


if __name__ == '__main__':
    from core.evaluation.film_release_evaluation import cal_acc
    from utils import read_json

    # model_size = '7b'
    model_size = '13b'
    print("model_size:",model_size)

    tp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/film/llama2-{model_size}-chat_on_film_release_untruthful_when_question_model_answer.json'
    tp_samples = read_json(tp_file)
    tp_acc = cal_acc(tp_samples, key='when_question_model_answer')

    # fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/iti/film_release/llama2-{model_size}-chat-iti_on_film_release_false_premise_question.json'
    fp_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/direct_prompt/film/llama2-{model_size}-chat_on_film_release_untruthful_why_fp_question_model_answer.json'
    fp_samples = read_json(fp_file)
    fp_acc = cal_acc(fp_samples, key='why_fp_question_model_answer')

    print("tp acc:",tp_acc)
    print("fp acc:",fp_acc)

    results = collect(tp_samples, fp_samples)
    for result in results:
        print(len(result))


    a = [s["why_fp_question_model_answer"] for s in results[1]]
